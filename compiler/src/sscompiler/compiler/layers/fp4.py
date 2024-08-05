import bitsandbytes.functional as F
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit

from ..conditions import linear_precondition
from .constants import *
from .quantized import QuantizedLayer


class Portable4BitLinear(Linear4bit, QuantizedLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        **kwargs,
    ):
        """
        Creates an empty layer. This is useful for quantization while loading the model
        """
        super(Portable4BitLinear, self).__init__(
            in_features,
            out_features,
            bias,
            **kwargs,
        )
        quant_type = kwargs.pop("quant_type", "nf4")
        compute_dtype = kwargs.pop("compute_dtype", torch.bfloat16)
        has_bias = kwargs.pop("bias", True)
        self.is_quantized = quant_type
        self.original_dtype = compute_dtype
        self.has_bias = bias

    @classmethod
    def empty_init(
        cls,
        in_features,
        out_features,
        bias,
        **kwargs,
    ):
        """
        Creates an empty layer. This is useful for quantization while loading the model
        """
        instance = cls(in_features, out_features, bias, **kwargs)
        Portable4BitLinear.__init__(
            instance,
            in_features,
            out_features,
            bias,
            **kwargs,
        )
        return instance

    @classmethod
    def from_linear(
        cls,
        existing_layer: nn.Linear,
        is_quantized="nf4",
    ):
        """
        Creates a quantized layer from an existing portable layer
        """
        assert is_quantized in [
            "fp4",
            "nf4",
        ], "Unsupported quantization type. Choose either 'fp4' or 'nf4'."

        existing_layer = linear_precondition(existing_layer)
        in_features, out_features = (
            existing_layer.in_features,
            existing_layer.out_features,
        )
        has_bias = existing_layer.bias is not None

        instance = cls(in_features, out_features, has_bias)
        Linear4bit.__init__(
            instance,
            input_features=in_features,
            output_features=out_features,
            bias=has_bias,
            quant_type=is_quantized,
        )

        instance.load_state_dict(
            state_dict=existing_layer.state_dict(),
            strict=False,
        )
        instance.original_dtype = existing_layer.weight.dtype
        del existing_layer

        # Quantization type (fp4 or nf4)
        if is_quantized == "fp4":
            instance.is_quantized = FP4
        else:
            instance.is_quantized = NF4
        instance.has_bias = has_bias
        return instance

    def dequantize(self) -> torch.Tensor:
        """
        Dequantizes the quantized tensor.

        Parameters:
        - A: torch.Tensor, the quantized tensor to be dequantized.
        """
        with torch.no_grad():
            if self.weight.quant_state is None:
                raise ValueError(
                    "Weight quantization state is not initialized. Please quantize before dequantizing."
                )

            # Use the existing quant_state for dequantization
            dequantized_w = F.dequantize_4bit(
                self.weight, quant_state=self.weight.quant_state
            )
            return dequantized_w

    def quantize(self, layer: nn.Linear):
        """
        Converts input layer to a Portable4BitLinear
        Takes in a linear layer and returns quantlized linear layer
        """
        return Portable4BitLinear(layer, self.is_quantized, self.has_bias).to(
            self.existing_layer.weight.device
        )
