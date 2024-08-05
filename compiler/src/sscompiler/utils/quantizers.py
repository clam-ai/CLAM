import torch


class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryInterface:
    def get_save_weight_dict(self):
        return {"weight": self.weight.data.half().cpu(), "bias": self.bias}
