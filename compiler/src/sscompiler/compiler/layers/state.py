import torch
from torch import nn

from .constants import *
from .quantized import QuantizedLayer


class OptimizationState:
    def __init__(self, existing_layer, in_features, out_features):
        self.existing_layer = existing_layer
        self.in_features = in_features
        self.out_features = out_features

        if isinstance(self.existing_layer, QuantizedLayer):
            self.quantize_function = self.existing_layer.quantize
        elif hasattr(self.existing_layer, "quantize_function"):
            self.quantize_function = self.existing_layer.quantize_function

        if isinstance(self.existing_layer, QuantizedLayer):
            self.ir_dtype = self.existing_layer.original_dtype
        else:
            assert self.existing_layer.weight is not None
            self.ir_dtype = self.existing_layer.weight.dtype

    def postprocess(self, altered_existing_layer: nn.Linear):
        new_layer = altered_existing_layer
        if hasattr(self, "quantize_function"):
            new_layer = self.quantize_function(altered_existing_layer).to(
                self.existing_layer.weight.device
            )
        return new_layer

    def get_weight(self) -> torch.Tensor:
        """
        Gets self.existing_layer.weight.data into a standard nn.Linear weight format
        """

        if isinstance(self.existing_layer, OptimizationState):
            x = self.existing_layer.get_equivalent_weight()
        elif isinstance(self.existing_layer, QuantizedLayer):
            x = self.existing_layer.dequantize()
        else:
            x = self.existing_layer.weight.data

        return self.convert_tensor_dtype(x)

    def get_bias(self) -> torch.Tensor:
        layer = self.existing_layer
        bias = None
        if isinstance(self.existing_layer, OptimizationState):
            bias = self.existing_layer.get_equivalent_bias()
        else:
            if self.existing_layer.bias is not None:
                bias = layer.bias.data
        if bias is not None:
            return self.convert_tensor_dtype(bias)
        else:
            return None

    def get_equivalent_layer(self):
        """
        Converts the current layer to a nn.Linear layer
        """
        weight = self.get_equivalent_weight()
        bias = self.get_equivalent_bias()
        with torch.no_grad():
            new_layer = nn.Linear(
                self.in_features,
                self.out_features,
                bias=bias is not None,
                dtype=weight.dtype,
            ).to(weight.device)

            # Copy weights
            new_layer.weight.data = weight.clone()

            if bias is not None:
                new_layer.bias.data = bias.clone()

            new_layer = self.postprocess(new_layer)

            return new_layer

    def convert_tensor_dtype(self, x: torch.Tensor):
        return x.to(self.ir_dtype)

    def get_equivalent_weight(self):
        raise NotImplementedError("Not implemented for this technique.")

    def get_equivalent_bias(self):
        raise NotImplementedError("Not implemented for this technique.")

    @property
    def weight(self):
        return self.get_equivalent_weight()

    @property
    def bias(self):
        return self.get_equivalent_bias()
