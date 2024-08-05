import torch
import torch.nn as nn

import gc
from ..data_utils import return_given_alpha
from .state import OptimizationState


class WandaLayer(nn.Module, OptimizationState):
    def __init__(
        self,
        layer: nn.Module,
        sparsity_ratio=0.5,
        prune_n=0,
        prune_m=0,
        use_variant=True,
        layer_id=0,
        layer_name="none",
    ):
        """
        Initializes the WandaLayer with specific properties and configurations.

        layer: The underlying PyTorch layer (e.g., nn.Linear) to be wrapped by WandaLayer.
        sparsity_ratio: Target sparsity ratio for pruning.
        prune_n: Number of weights to keep in each group for structured pruning.
        prune_m: Group size for structured pruning.
        use_variant: Whether to use a pruning variant not mentioned in the paper.
        layer_id: Identifier for the layer, useful for models with multiple layers.
        layer_name: Human-readable name for the layer.
        """
        # check if layer type is allowed
        # from .constants import ALLOWED_LAYER_TYPES
        # allowed_types = ALLOWED_LAYER_TYPES.get(WandaLayer, [])
        # if not isinstance(layer, tuple(ALLOWED_LAYER_TYPES.get(self.__class__, []))):
        #     raise TypeError(f"layer must be an instance of one of the following: {ALLOWED_LAYER_TYPES.get(self.__class__, [])}, got {type(layer)}")

        nn.Module.__init__(self)
        OptimizationState.__init__(self, layer, layer.in_features, layer.out_features)

        self.in_features = self.existing_layer.in_features
        self.out_features = self.existing_layer.out_features
        self.columns = self.in_features
        self.dev = getattr(self.existing_layer, "weight.device", "cuda")

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.sparsity_ratio = sparsity_ratio
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.use_variant = use_variant

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
        Adjusts the scaling row (input activation size per row) based on the input batch for use in calculating pruning metrics.

        inp: Input tensor to the layer.
        out: Output from the layer
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.existing_layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32).to(self.scaler_row.device)
        # the scalar row is the l2 norm of the input to a layer; will be a key factor in pruning
        self.scaler_row += (torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples).squeeze()

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        return self.existing_layer(x)

    def prune(self):
        """
        Prunes the layer based on configured sparsity parameters and measured input statistics.
        Applies either structured or unstructured pruning based on the configuration.
        """
        preprocessed_weight = self.get_weight().to(self.scaler_row.device)
        W_metric = torch.abs(preprocessed_weight) * torch.sqrt(
            self.scaler_row.reshape((1, -1))
        )

        W_mask = torch.zeros_like(W_metric) == 1  ## initialize a mask to be all False
        if self.prune_n != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % self.prune_m == 0:
                    tmp = W_metric[:, ii : (ii + self.prune_m)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1],
                        True,
                    )
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            if self.use_variant:
                # wanda variant that is not mentioned in paper but provided in code
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)

                alpha = 0.4
                alpha_hist = [0.0, 0.8]
                W_mask, cur_sparsity = return_given_alpha(
                    alpha, sort_res, W_metric, tmp_metric, sum_before
                )
                while (torch.abs(cur_sparsity - self.sparsity_ratio) > 0.001) and (
                    alpha_hist[1] - alpha_hist[0] >= 0.001
                ):
                    if cur_sparsity > self.sparsity_ratio:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha

                    alpha = alpha_new
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                # print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
            else:
                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

        preprocessed_weight[W_mask.view(-1, preprocessed_weight.shape[1])] = (
            0  # prunes matrix
        )

        # new linear layer
        bias = self.get_bias()
        new_layer = nn.Linear(
            self.in_features, self.out_features, bias=bias is not None
        )
        new_layer.weight.data = preprocessed_weight
        if bias is not None:
            new_layer.bias.data = bias

        new_linear = self.postprocess(new_layer)
        dev = self.existing_layer.weight.device
        del self.existing_layer
        self.existing_layer = new_linear.to(dev)

        # garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    def get_layer(self):
        return self.existing_layer

    def get_equivalent_weight(self):
        return self.get_weight()

    def get_equivalent_bias(self):
        return self.get_bias()
