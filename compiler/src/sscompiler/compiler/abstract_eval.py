import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from .abstract import AbstractTransformer


@register_model("abstract")
class AbstractLMForEval(HFLM):
    def __init__(
        self,
        base_model,
        dtype=torch.bfloat16,
        device="cuda",
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=base_model,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def load_abstract_transformer(self, at: AbstractTransformer):
        if isinstance(at, AbstractTransformer):
            self._model = at.auto_model
        else:
            raise TypeError("not an AbstractTransformer")
