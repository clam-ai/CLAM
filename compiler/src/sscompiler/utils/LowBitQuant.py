import numpy as np
import torch


class LowBitQuant:
    def __init__(self, num_bits, dtype):
        # initialize variables
        self.salient_compressed = None
        self.binary_scales = None

        # how many weights fit into a byte
        self.weights_per_byte = 8 // num_bits
        # num bits per weight to quantize to
        self.num_bits = num_bits

        # Storage of pos/neg binarized weights
        self.compressed = None

        # Save mask and mean
        self.mean = None

        # save dtype
        self.dtype = dtype
        self.mask = None

    @torch.no_grad
    def compression(self, weight, binary_scales, means, salients, mask) -> None:
        """
        Method to compress weights
        weight: Binarized weight matrix w/o scale applied w/o salient weights
        mask: True where not salient weights
        salients: a sparse matrix with just salient weights
        """
        # Save Binary Scales and mask and mean
        self.binary_scales = binary_scales
        self.mean = means

        # convert to numpy aray
        weight_np = weight.to(torch.uint8).cpu().detach().numpy()
        self.compressed = np.packbits(weight_np, axis=1)

        # save salient weights
        self.salient_compressed = salients.to_sparse()
        # self.mask = (~mask).to_sparse()
        self.mask = (mask).cpu().detach().numpy()
        self.mask = np.packbits(self.mask, axis=1)
        # we can do pack bits for mask
        del weight
        del salients
        del mask
        torch.cuda.empty_cache()

    @torch.no_grad
    def decompress(self) -> torch.Tensor:
        decomp = np.unpackbits(self.compressed, axis=1)
        decomp = torch.from_numpy(decomp).to(self.dtype).to(device="cuda")
        # add negative and restore binarization
        decomp = torch.where(decomp < 1, -1, decomp)
        decomp *= self.binary_scales
        decomp += self.mean
        # restore salient weights
        salient_uncompr = self.salient_compressed.to_dense()
        mask_uncompr = torch.from_numpy(
            np.unpackbits(self.mask, axis=1).astype(bool)
        ).to(device="cuda")
        res = salient_uncompr * ~mask_uncompr + decomp * mask_uncompr

        return res
