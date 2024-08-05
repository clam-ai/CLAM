from torch import nn


def mark_all_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = True
