import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import OptimizationState
from .peft import PEFTAdapter
import math

from typing import Optional


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: Optional[int],
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        if lora_alpha is None:
            self.lora_alpha = r if r > 0 else 1
        else:
            self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class PortableLoRAAdapter(PEFTAdapter, LoRALayer):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        lora_lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__(existing_layer, in_features, out_features)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
        )
        self.lora_lr = lora_lr
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros((r, self.in_features), dtype=self.ir_dtype, device="cuda")
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, r), dtype=self.ir_dtype, device="cuda")
            )
            # Use rsLoRA
            self.scaling = self.lora_alpha / math.sqrt(self.r)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.linear(
                self.lora_dropout(x),
                self.get_equivalent_weight(),
                self.get_equivalent_bias(),
            )
        else:
            return self.existing_layer.forward(x)

    def get_equivalent_weight(self):
        """
        Converts LoRA layer to equivalent nn.Linear weight tensor
        """
        converted_weight = self.get_weight()
        if self.r > 0:
            return converted_weight + (self.lora_B @ self.lora_A) * self.scaling
        else:
            return converted_weight

    def get_equivalent_bias(self):
        """
        Gets equivalent nn.Linear bias data
        """
        converted_bias = self.get_bias()
        return converted_bias

    def get_params_lr(self):
        """
        Returns a list of dictionaries, each containing parameters and their associated learning rate.
        """
        return [{"params": [self.lora_A, self.lora_B], "lr": self.lora_lr}]
