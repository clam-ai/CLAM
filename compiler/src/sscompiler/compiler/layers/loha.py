import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import OptimizationState
from .peft import PEFTAdapter
import math


class LoHALayer:
    def __init__(
        self,
        r: int,
        loha_alpha: int,
    ):
        self.r = r
        self.loha_alpha = loha_alpha


def mark_only_loha_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "loha_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "loha_only":
        for m in model.modules():
            if isinstance(m, LoHALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class PortableLoHAAdapter(PEFTAdapter, LoHALayer):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        r: int = 0,
        loha_alpha: int = 1,
        loha_lr: float = 2e-4,
        **kwargs,
    ):
        super().__init__(existing_layer, in_features, out_features)
        LoHALayer.__init__(
            self,
            r=r,
            loha_alpha=loha_alpha,
        )
        self.loha_lr = loha_lr
        # Actual trainable parameters
        if r > 0:
            self.loha_A1 = nn.Parameter(
                torch.zeros((r, self.in_features), dtype=self.ir_dtype, device="cuda")
            )
            self.loha_B1 = nn.Parameter(
                torch.zeros((self.out_features, r), dtype=self.ir_dtype, device="cuda")
            )
            self.loha_A2 = nn.Parameter(
                torch.zeros((r, self.in_features), dtype=self.ir_dtype, device="cuda")
            )
            self.loha_B2 = nn.Parameter(
                torch.zeros((self.out_features, r), dtype=self.ir_dtype, device="cuda")
            )
            # Use rsLoRA
            self.scaling = self.loha_alpha / math.sqrt(self.r)
            # Freezing the pre-trained weight matrix
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "loha_A1"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.loha_A1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.loha_A2, a=math.sqrt(5))
            # nn.init.zeros_(self.loha_B1)
            # nn.init.zeros_(self.loha_B2)
            nn.init.kaiming_uniform_(self.loha_B1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.loha_B2, a=math.sqrt(5))

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
        """
        Converts LoRA layer to equivalent nn.Linear weight tensor
        """
        converted_weight = self.get_weight()
        if self.r > 0:
            return (
                converted_weight
                + (self.loha_B1 @ self.loha_A1)
                * (self.loha_B2 @ self.loha_A2)
                * self.scaling
            )
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
        return [
            {
                "params": [self.loha_A1, self.loha_B1, self.loha_A2, self.loha_B2],
                "lr": self.loha_lr,
            }
        ]
