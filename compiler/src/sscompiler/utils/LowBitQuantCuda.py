import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from .constants import BASE_DIR

# Load the custom CUDA extension
low_bit_quant = load(
    name="low_bit_quant",
    sources=[
        os.path.join(BASE_DIR, "compiler/src/sscompiler/utils/low_bit_quant_kernel.cu")
    ],
    verbose=True,
)


class LowBitQuantCUDA(nn.Module):
    salient_compressed: torch.Tensor | None
    salient_scale: torch.Tensor | None
    binary_scales: torch.Tensor | None
    compressed: torch.Tensor | None
    mean: torch.Tensor | None
    mask: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights_per_byte = 8 // num_bits
        self.num_bits = num_bits
        self.dtype = dtype

        self.salient_compressed = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        self.binary_scales = torch.nn.Parameter(
            torch.empty(out_features, 1, dtype=self.dtype),
            requires_grad=False,
        )
        self.compressed = torch.nn.Parameter(
            torch.empty(out_features, in_features // 8, dtype=torch.uint8),
            requires_grad=False,
        )
        self.mean = torch.nn.Parameter(
            torch.empty(out_features, 1, dtype=self.dtype), requires_grad=False
        )
        self.mask = torch.nn.Parameter(
            torch.empty(out_features, in_features // 8, dtype=torch.uint8),
            requires_grad=False,
        )
        self.salient_scale = torch.nn.Parameter(
            torch.empty(out_features, 1, dtype=self.dtype),
            requires_grad=False,
        )
        self.salient_zero = torch.nn.Parameter(
            torch.empty(out_features, 1, dtype=self.dtype),
            requires_grad=False,
        )

    @torch.no_grad()
    def compression(
        self,
        weight,
        binary_scales,
        means,
        salients,
        mask,
        salient_zero,
        salient_scale,
        datatype,
    ):
        """
        Compress full matricies and save any scales/means necessary to recontrusct the original matrix
        """

        self.dtype = datatype
        # pack weight and masks
        weight_uint8 = weight.to(torch.uint8)
        mask_uint8 = mask.to(torch.uint8)

        self.salient_compressed = torch.nn.Parameter(
            salients.to(torch.uint8).to_sparse_csr(), requires_grad=False
        )
        self.binary_scales = torch.nn.Parameter(
            binary_scales.to(self.dtype), requires_grad=False
        )
        self.compressed = torch.nn.Parameter(
            low_bit_quant.packbits_cuda(weight_uint8),
            requires_grad=False,
        )
        self.mean = torch.nn.Parameter(means.to(self.dtype), requires_grad=False)
        self.mask = torch.nn.Parameter(
            low_bit_quant.packbits_cuda(mask_uint8), requires_grad=False
        )
        self.salient_scale = torch.nn.Parameter(
            salient_scale.to(self.dtype), requires_grad=False
        )
        self.salient_zero = torch.nn.Parameter(
            salient_zero.to(self.dtype), requires_grad=False
        )

        # cleanup
        weight = None
        binary_scales = None
        means = None
        salients = None
        mask = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def decompress(self) -> torch.Tensor:
        """
        reverse compression and restore original weights
        """
        # building binarized matrix
        rows, cols = self.compressed.size(0), self.compressed.size(1) * 8
        decomp_uint8 = low_bit_quant.unpackbits_cuda(self.compressed, rows, cols)
        decomp = decomp_uint8.to(self.dtype)

        decomp = torch.where(decomp < 1, -1, decomp)
        decomp *= self.binary_scales
        decomp += self.mean
        # restoring salient matrix
        salient_uncompr = self.salient_compressed.to_dense()
        salient_uncompr = self.salient_scale * (salient_uncompr - self.salient_zero)
        salient_uncompr = salient_uncompr.to(self.dtype)
        rows, cols = self.mask.size(0), self.mask.size(1) * 8
        # restore mask
        mask_uncompr = low_bit_quant.unpackbits_cuda(self.mask, rows, cols).to(
            torch.bool
        )
        # puts salient and binarized together
        res = salient_uncompr * ~mask_uncompr + decomp * mask_uncompr
        return res
