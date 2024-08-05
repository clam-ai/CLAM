import torch
import torch.nn as nn

from .peft import PEFTAdapter


class PortableIA3Adapter(PEFTAdapter):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        existing_layer: nn.Linear,
        in_features,
        out_features,
        is_feedforward=False,
        ia3_lr: float = 3e-3,
        **kwargs,
    ):
        super().__init__(existing_layer, in_features, out_features)
        self.is_feedforward = is_feedforward
        self.ia3_lw = (
            nn.Parameter(
                torch.ones((1, out_features), dtype=self.ir_dtype, device="cuda")
            )
            if not is_feedforward
            else nn.Parameter(
                torch.ones((1, in_features), dtype=self.ir_dtype, device="cuda")
            )
        )
        self.ia3_lr = ia3_lr

        nn.init.ones_(self.ia3_lw)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        # TODO Make IA3 forward avoid calling existing_layer's forward
        if not self.is_feedforward:
            result = self.existing_layer.forward(x)
            result = torch.mul(result, self.ia3_lw)
            return result
        else:
            result = torch.mul(x, self.ia3_lw)
            result = self.existing_layer(result)
            return result

    def get_equivalent_weight(self):
        """
        Converts IA3 layer to equivalent nn.Linear weight tensor
        """
        mat = self.get_weight()
        ret_weight = None
        if not self.is_feedforward:
            # ret_weight = torch.mul(mat.T, self.ia3_lw.view(-1))
            ret_weight = torch.diag(self.ia3_lw.view(-1)) @ mat
        else:
            ret_weight = mat @ torch.diag(self.ia3_lw.view(-1))

        return ret_weight

    def get_equivalent_bias(self):
        """
        Gets equivalent nn.Linear bias data
        """
        mat = self.get_bias()
        if mat is None:
            return None
        ret_bias = None

        if not self.is_feedforward:
            ret_bias = torch.mul(mat, self.ia3_lw.squeeze())
        else:
            ret_bias = mat

        return ret_bias

    def get_params_lr(self):
        """
        Returns a list of dictionaries, each containing parameters and their associated learning rate.
        """
        return [{"params": [self.ia3_lw], "lr": self.ia3_lr}]


def mark_only_ia3_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "ia3_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
