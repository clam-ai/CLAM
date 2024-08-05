import bitsandbytes as bnb
import bitsandbytes.functional as F
import torch
from torch import nn

from .constants import *
from .state import OptimizationState


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    Returns a low rank decomposition of the input weight matrix with the
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(
            f"Only support 2D matrix, but your input has {matrix_dimension} dimensions."
        )

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh
    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh}


class PortableLoftQLayer(nn.Module, OptimizationState):
    def __init__(
        self,
        existing_layer: nn.Module,
        num_bits: int,
        r: int,
        num_iter: int = 1,
        loftq_lr: float = 3e-4,
    ):
        nn.Module.__init__(self)
        # TODO: jank way of doing things
        self.original_dtype = existing_layer.weight.dtype
        OptimizationState.__init__(
            self,
            existing_layer,
            existing_layer.in_features,
            existing_layer.out_features,
        )
        A = torch.zeros(
            (existing_layer.in_features, r), dtype=self.ir_dtype, device="cuda"
        )
        B = torch.zeros(
            (r, existing_layer.out_features), dtype=self.ir_dtype, device="cuda"
        )
        self.num_bits = num_bits
        self.quant_type = "nf4"
        self.has_bias = self.existing_layer.bias is not None
        qweight, lora_A, lora_B = _loftq_weights(
            self.existing_layer, A, B, num_bits, r, num_iter
        )
        lora_A = lora_A.to(self.ir_dtype)
        lora_B = lora_B.to(self.ir_dtype)
        # self.load_state_dict(existing_layer.state_dict(), strict=False)
        self.existing_layer.weight = qweight
        self.quant_state = qweight.quant_state
        self.is_quantized = NF4
        self.quant_type = "nf4"

        # add nn.Parameter
        self.lora_A = nn.Parameter(lora_A.T)
        self.lora_B = nn.Parameter(lora_B.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO make forward pass work better with invariants
        result = bnb.matmul_4bit(
            x, self.existing_layer.weight, quant_state=self.quant_state
        )
        torch_result_dtype = result.dtype
        x = x.to(self.lora_A.dtype)
        if self.has_bias:
            bias = self.existing_layer.bias.to(self.lora_A.dtype)
        else:
            bias = None
        lora_result = torch.nn.functional.linear(
            x,
            self.lora_B @ self.lora_A,
            bias,
        )
        result = result + lora_result
        result = result.to(torch_result_dtype)
        return result

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def quantize(self, A: torch.Tensor, compress_statistics=False) -> torch.Tensor:
        quantized_tensor, _ = F.quantize_4bit(
            A,
            blocksize=self.quant_state.blocksize,
            compress_statistics=compress_statistics,
            quant_type="nf4",
        )
        return quantized_tensor

    def dequantize(self) -> nn.Linear:
        with torch.no_grad():
            dequantized_w = F.dequantize_4bit(
                self.existing_layer.weight,
                quant_state=self.quant_state,
                quant_type="nf4",
            ).to(self.lora_A.dtype)
            linear = nn.Linear(
                in_features=dequantized_w.shape[1],
                out_features=dequantized_w.shape[0],
                bias=self.has_bias,
            ).to(self.lora_A.dtype)
            linear.weight.data = dequantized_w
            if self.has_bias:
                linear.bias.data = self.existing_layer.bias.data.to(self.lora_A.dtype)
            return linear

    def get_equivalent_weight(self):
        dequantized = self.dequantize().weight.data

        ret_weight = dequantized.T + self.lora_B @ self.lora_A
        return ret_weight.to(self.lora_A.dtype)

    def get_equivalent_bias(self):
        bias = self.get_bias()
        return bias

    def get_params_lr(self):
        """
        Returns a list of dictionaries, each containing parameters and their associated learning rate.
        """
        return [{"params": [self.lora_A, self.lora_B], "lr": self.loftq_lr}]


@torch.no_grad()
def _loftq_weights(module: nn.Module, A, B, num_bits: int, r: int, num_iter=1):
    if num_bits not in [4]:
        raise ValueError("Only support 4 bits quantization")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")

    compute_device = "cuda:0"
    with torch.no_grad():
        W = (
            module.weight.clone()
            .detach()
            .to(device=compute_device, dtype=torch.float32)
            .T
        )
        for _ in range(num_iter):
            torch.cuda.empty_cache()
            Q_t = W - A @ B
            qweight = bnb.nn.Params4bit(
                Q_t.to("cpu"),
                requires_grad=False,
                compress_statistics=False,
                quant_type="nf4",
            ).to(compute_device)
            dequantized_weight = F.dequantize_4bit(
                A=qweight.data, quant_state=qweight.quant_state
            )
            res = W - dequantized_weight
            output = _low_rank_decomposition(res, reduced_rank=r)
            A, B = output["L"], output["R"]

        return qweight, A, B
