""""""

import abc

import torch


class QuantizedLayer:
    """
    A common interface for quantized layers in the CLAM framework
    """

    original_dtype: "torch.dtype"

    @classmethod
    @abc.abstractmethod
    def from_linear(cls, *args, **kwargs):
        """
        Function for converting an existing linear layer into a quantized layer
        """

    @classmethod
    @abc.abstractmethod
    def empty_init(cls, *args, **kwargs):
        """
        Helper function for creating a quantized layer instance with no data inside.
        Particularly useful for quantizing models "on-the-fly"
        """

    @abc.abstractmethod
    def quantize(self, layer: torch.nn.Linear):
        """
        Defines how to quantize a linear layer according to the method being implemented
        """

    @abc.abstractmethod
    def dequantize(self) -> torch.Tensor:
        """
        Function for dequantizing a layer into a torch.Tensor of the dequantized weight
        """
