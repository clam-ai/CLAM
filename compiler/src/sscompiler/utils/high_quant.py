import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function


@torch.no_grad
class HighQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        in_features,
        out_features,
        perchannel=True,
        sym=False,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        grouprows=1,
        shape=1,
    ):
        super().__init__()

        self.maxq = torch.tensor(2**bits - 1)

        # this will need to be modified to be conditional on grouprows and perchannel
        self.register_buffer("scale", torch.zeros([out_features, 1]))
        self.register_buffer("zero", torch.zeros([out_features, 1]))

        self.perchannel = perchannel
        self.grouprows = grouprows

    # calibration to generate zero-point and scale factor for weights
    def calibrate(self, x, weight=True):
        dev = x.device
        maxq = self.maxq

        maxq = maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1:
                    x = x.reshape((x.shape[0] // self.grouprows, -1))

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq

        zero = torch.round(-xmin / scale)

        x = None
        if weight:
            if self.grouprows > 1:
                scale = scale.unsqueeze(1).repeat(1, self.grouprows)
                zero = zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)

        self.scale = scale
        self.zero = zero

    def quantize(self, x, blocki=None):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)

        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

    def get_zero(self):
        return self.zero

    def get_scale(self):
        return self.scale

    def free(self):
        self.scale = None
        self.maxq = None
        self.zero = None


@torch.no_grad
def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)  # this is in 8 bit
    # return raw 8bit quantized as well as dequantized weights
    return q, dequantize(q, scale, zero)


@torch.no_grad
def dequantize(x, scale, zero):
    return scale * (x - zero)
