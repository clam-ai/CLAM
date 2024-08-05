import gc
import math
import time

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.utils.quantization_config import (
    QuantizationConfigMixin,
    QuantizationMethod,
)

from ...utils.high_quant import HighQuantizer
from ...utils.LowBitQuantCuda import LowBitQuantCUDA
from ...utils.quantizers import STEBinary
from .quantized import QuantizedLayer


# A quicker quantization method for QAT
def weight_quant_8bit(w, simulated=True):
    raw_type = w.dtype
    # per channel assymetric quantization
    w_range = (
        torch.max(w, dim=-1, keepdim=True)[0] - torch.min(w, dim=-1, keepdim=True)[0]
    )
    w_range = w_range.type(torch.float32)  # x_max - x_min per channel
    w_zero_point = torch.round(torch.min(w, dim=-1, keepdim=True)[0])

    w_q = torch.round((w - w_zero_point) / w_range * 255).type(torch.uint8)
    # clip
    w_q = torch.clamp(w_q, 0, 255)
    if simulated:
        # dequantize
        w_q = w_q * (w_range / 255) + w_zero_point
        w_q = w_q.to(raw_type)
    else:
        w_q = w_q.type(torch.uint8)
    return w_q


class PBLLMConfig(QuantizationConfigMixin):
    """Config class for PBLLM layers."""

    quant_method: QuantizationMethod = QuantizationMethod.PBLLM

    def __init__(
        self,
        low_frac: float,
        original_dtype: torch.dtype,
        num_bits: int = 1,
        mode: str = "QAT",
    ) -> None:
        self.low_frac = low_frac
        self.original_dtype = original_dtype
        self.num_bits = num_bits
        self.mode = mode


class GPTQLayer(QuantizedLayer, nn.Module):
    """Quantized PBLLM layer"""

    original_dtype: torch.dtype | None
    num_bits: int
    low_frac: float
    mode: str
    dev: torch.device
    mask: torch.Tensor
    H: torch.Tensor
    high_quantizer: HighQuantizer
    lbq: LowBitQuantCUDA
    existing_layer: nn.Module
    quantized: bool
    bias: torch.Tensor | None

    def __init__(self, in_features, out_features, bias, original_dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.original_dtype = original_dtype
        self.quantized = False

    @classmethod
    def from_linear(
        cls,
        *args,
        **kwargs,
    ):
        """Creates a PBLLM quantized layer from a pre-existing, full-precision, linear layer."""
        layer, num_bits, low_frac = args
        mode = kwargs.get("mode", "PTQ")
        assert mode in ["QAT", "PTQ"], "Invalid mode for PBLLM layer"
        in_features, out_features = layer.in_features, layer.out_features
        instance = cls(
            in_features,
            out_features,
            layer.bias,
            layer.weight.dtype,
        )

        # Initialize the parent classes
        nn.Module.__init__(instance)

        # store instance variables
        instance.num_bits = num_bits
        instance.low_frac = low_frac
        instance.mode = mode
        instance.existing_layer = layer

        # Store the parameters

        instance.dev = layer.weight.device
        instance.mask = None

        # Initialize the quantizers (for PTQ)
        if instance.mode == "PTQ":
            high_quantizer = HighQuantizer(num_bits, in_features, out_features)
            high_quantizer.calibrate(instance.existing_layer.weight)
            instance.high_quantizer = high_quantizer

        # Initialize Low Bit Quantizer for 1 bit packing
        instance.lbq = LowBitQuantCUDA(in_features, out_features, 1, layer.weight.dtype)

        return instance

    def createH(self):
        self.H = torch.zeros(
            (self.existing_layer.in_features, self.existing_layer.in_features),
            device=self.existing_layer.weight.device,
        )

    @classmethod
    def empty_init(
        cls,
        *args,
        **kwargs,
    ):
        """Create a shallow instance of a PBLLM layer for filling in a meta model"""
        # TODO: add PBLMM config class for instantiating layers can store this
        # information in the model's config so that it can be easily retrieved
        # from models saved to disc
        q_config: PBLLMConfig = kwargs.pop("quantization_config")

        in_features, out_features, bias = args
        if bias:
            bias = torch.empty(out_features, 1)
        else:
            bias = None

        assert not isinstance(bias, bool), "somehow did not get converted :D"

        instance = cls(
            in_features,
            out_features,
            bias,
            original_dtype=q_config.original_dtype,
            **kwargs,
        )

        instance.num_bits = q_config.num_bits
        instance.low_frac = q_config.low_frac
        instance.mode = q_config.mode

        instance.lbq = LowBitQuantCUDA(
            instance.in_features,
            instance.out_features,
            1,
            q_config.original_dtype,
        )
        instance.quantized = True

        return instance

    def create_H(self) -> None:
        self.register_buffer(
            "H", torch.zeros((self.in_features, self.in_features), device=self.dev)
        )

    @torch.no_grad
    def gen_mask(
        self,
        H: torch.Tensor | None = None,
    ) -> None:
        """Generate a mask of binarized weights"""
        W = self.existing_layer.weight.data.clone()
        if isinstance(self.existing_layer, transformers.Conv1D):
            W = W.t()

        if H is not None:
            # hessian criteria
            saliency = W**2 / (torch.diag(H).reshape((1, -1))) ** 2
            thresh = torch.sort(saliency.flatten())[0][
                int(saliency.numel() * self.low_frac)
            ]
        else:
            # magnitude criteria
            saliency = torch.abs(W)
            thresh = torch.sort(saliency.flatten())[0][
                int(saliency.numel() * self.low_frac)
            ]
            self.existing_layer.weight.data = weight_quant_8bit(W)

        # a mask is created where weights are less than the saliency threshold
        self.mask = saliency <= thresh
        W = None

    @torch.no_grad
    def quantize(
        self,
        percdamp: float = 0.01,
        blocksize: int = 128,
    ) -> None:
        # Clone the weight data to avoid modifying the original weights
        W = self.existing_layer.weight.data

        # quantize in float32 format
        W = W.float()

        # If the layer is a Conv1D layer from the transformers library, transpose W
        if isinstance(self.existing_layer, transformers.Conv1D):
            W = W.t()

        # Number of columns in the weight matrix
        n_cols = W.shape[1]

        # For numerical stability
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        # set non-diagonal elements to 0
        W[:, dead] = 0

        # Calculate a damping factor (number to add to each diagonal element)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(n_cols, device=self.dev)
        H[diag, diag] += damp

        # finds upper trianglular matrix
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # generate mask
        if self.mask is None:
            self.gen_mask(H)

        # calculate optimal binary scale factor - analogous to low_quantizer.calibrate
        masked = W * self.mask
        w_mean = (masked).mean(-1).view(-1, 1)
        masked = masked - w_mean
        binary_scale = (masked).abs().mean(-1, keepdim=True)

        # Initialize matrices for losses
        Losses = torch.zeros(W.shape[0])

        # seperate matrix to pass into one bit compressor
        W_clone = torch.zeros_like(W)

        tick = time.time()
        # actual quantization begins, does it with blocks of blocksize
        for blocki, col_st in enumerate(range(0, n_cols, blocksize)):
            col_ed = min(col_st + blocksize, n_cols)
            num_cols = col_ed - col_st

            # generate some utility matricies
            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Q1_clone = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
            Losses1 = torch.zeros_like(W1)
            mask1 = self.mask[:, col_st:col_ed]

            # Iterate over each column of the block
            for i in range(num_cols):
                # Extract the i-th column of weights
                w = W1[:, i]

                # Extract the i-th diagonal element of Hinv
                d = Hinv1[i, i]

                # quantize salient weights
                q_high_quant, q_high = self.high_quantizer.quantize(w.unsqueeze(1))
                q_high = q_high.flatten()
                q_high_quant = q_high_quant.flatten()

                # use mean as zero point to binarize
                q_low = w.unsqueeze(1)
                q_low -= w_mean

                q_low = torch.sign(q_low)
                q_clone = torch.where(q_low <= 0, 0, 1)

                # puts binarized non-salient and raw quantized salient weights together (to pass into compression)
                q_clone = (
                    q_clone.flatten().squeeze() * mask1[:, i]
                    + q_high_quant * ~mask1[:, i]
                )

                # applies scale and mean binarized weights
                q_low = q_low * binary_scale
                q_low += w_mean
                q_low = q_low.flatten()

                # Combine high and low quantized weights using the mask
                q = q_high * ~mask1[:, i] + q_low.squeeze() * mask1[:, i]

                # Store the quantized weights
                Q1[:, i] = q
                Q1_clone[:, i] = q_clone

                # Calculate the loss for the current column
                Losses1[:, i] = ((w - q) ** 2) / d**2

                # Calculate the quantization error and adjust the remaining weights
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            # put quantized block back into Q1
            W[:, col_st:col_ed] = Q1
            W_clone[:, col_st:col_ed] = Q1_clone
            Losses += torch.sum(Losses1, 1).to("cpu") / 2
            # global error adjustment
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

        torch.cuda.synchronize()

        # 1 bit compression
        self.lbq.compression(
            W_clone * self.mask,
            binary_scale,
            w_mean,
            W_clone * ~self.mask,
            self.mask,
            self.high_quantizer.get_zero(),
            self.high_quantizer.get_scale(),
            self.existing_layer.weight.dtype,
        )

        # free local variables
        self.high_quantizer.free()
        self.high_quantizer = None
        W = None
        H = None
        binary_scale = None
        w_mean = None
        W_clone = None
        self.mask = True
        self.quantized = True
        del self.existing_layer.weight
        torch.cuda.empty_cache()
        gc.collect()

    def binarizeExceptSalient(self):
        # method for binarization doing QAT
        if self.mask is None:
            self.gen_mask()
        W = self.existing_layer.weight.data.clone().to(self.dev)
        Q1 = torch.zeros_like(W)

        # Apply binarization and STE to weights and return binarized matrix
        binary_scale = self.existing_layer.weight[self.mask].abs().mean(-1).view(-1, 1)
        binary_weight = STEBinary().apply(W) * binary_scale
        Q1 = torch.where(self.mask, binary_weight, W)

        return Q1

    def train(self, mode: bool = True):
        if hasattr(self, "existing_layer"):
            self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        # do binarized forward pass here
        if not self.quantized:
            return self.existing_layer(x)
        else:
            # quantize and pass data
            return F.linear(
                x,
                self.dequantize(),
                self.bias,
            )

    # this should be refactored slightly. we should not retain the existing_layer at all
    def dequantize(self):
        if self.mode == "PTQ":
            return self.lbq.decompress()
        else:
            return self.binarizeExceptSalient()

    def get_params_lr(self):
        """
        Returns a list of dictionaries, each containing parameters and their associated learning rate.
        """
        return [{"params": [self.existing_layer.weight], "lr": 2e-5}]

    def add_batch(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        blocksize: int = 1024,
        nsamples: int = 128,
    ):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.existing_layer, nn.Linear) or isinstance(
            self.existing_layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= nsamples / (nsamples + tmp)
        nsamples += tmp
        inp = math.sqrt(2 / nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def free(self):
        """Frees memory of weights that are not used after quantization"""
        if hasattr(self, "H"):
            del self.H
        torch.cuda.empty_cache()
