import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import OptimizationState
from .peft import PEFTAdapter
import math
from typing import Optional


class PortableVeRAAdapter(PEFTAdapter, nn.Module):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        r: int = 0,
        d_initial: float = 0.1,
        vera_lr: float = 1e-2,
    ):
        self.r = r
        self.d_initial = d_initial
        super().__init__(existing_layer, in_features, out_features)
        # Actual trainable parameter
        self.vera_lr = vera_lr
        original_device = next(existing_layer.parameters()).device
        self.A = torch.randn(
            (r, in_features), dtype=self.ir_dtype, device=original_device
        )
        self.B = torch.randn(
            (out_features, r), dtype=self.ir_dtype, device=original_device
        )
        self.vera_lambda_b = nn.Parameter(
            torch.ones(
                (self.out_features, 1),
                requires_grad=True,
                dtype=self.ir_dtype,
                device=original_device,
            )
        )
        self.vera_lambda_d = nn.Parameter(
            torch.randn(
                (r, 1), requires_grad=True, dtype=self.ir_dtype, device=original_device
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.vera_lambda_d, self.d_initial)
        nn.init.zeros_(self.vera_lambda_b)
        torch.nn.init.xavier_normal_(self.A)
        torch.nn.init.xavier_normal_(self.B)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            return F.linear(
                x,
                self.get_equivalent_weight(),
                self.get_equivalent_bias(),
            )
        else:
            return self.existing_layer.forward(x)

    def get_equivalent_weight(self):
        converted_weight = self.get_weight()
        if self.r > 0:
            return converted_weight + (
                (self.vera_lambda_b * self.B) @ (self.vera_lambda_d * self.A)
            )
        else:
            return converted_weight

    def get_equivalent_bias(self):
        converted_bias = self.get_bias()
        return converted_bias

    def get_params_lr(self):
        """
        Retrieves parameters and their associated learning rates.
        """
        return [
            {"params": [self.vera_lambda_b, self.vera_lambda_d], "lr": self.vera_lr}
        ]


def mark_only_vera_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "vera_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
