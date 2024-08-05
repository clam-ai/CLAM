import torch.nn as nn
from .state import OptimizationState
from .peft import PEFTAdapter


class BitFit(PEFTAdapter):
    def __init__(self, existing_layer: nn.Module, bitfit_lr: float = 1e-4):
        super().__init__(
            existing_layer, existing_layer.in_features, existing_layer.out_features
        )
        self.enable_bias_gradients_only()
        self.bitfit_lr = bitfit_lr

    def forward(self, x):
        return self.existing_layer(x)

    def enable_bias_gradients_only(self):
        for name, param in self.existing_layer.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_equivalent_weight(self):
        return self.existing_layer.weight.data

    def get_equivalent_bias(self):
        if self.existing_layer.bias is not None:
            return self.existing_layer.bias.data
        return None

    def get_params_lr(self):
        """Return all parameters of the existing layer with bitfit_lr if they require gradients."""
        params_lr = []
        for name, param in self.existing_layer.named_parameters():
            if param.requires_grad:
                # Only return parameters that require gradients with the associated learning rate
                params_lr.append({"params": [param], "lr": self.bitfit_lr})
        return params_lr
