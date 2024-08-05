import torch.nn as nn
from .state import OptimizationState


class PEFTAdapter(nn.Module, OptimizationState):
    def __init__(self, existing_layer: nn.Module, in_features, out_features):
        super().__init__()
        OptimizationState.__init__(self, existing_layer, in_features, out_features)


def mark_adapters_as_trainable(model: nn.Module, bias: str = "none"):
    # Make adapter-specific parameters require gradient
    for name, param in model.named_parameters():
        param.requires_grad = False

        # if is_param_in_bitfit_module(model, param) and "bias" in name:
        #     param.requires_grad = True
        # else:
        if (
            "lora_" in name or "ia3_" in name or "vera_" in name or "loha_" in name
        ):  # or "classification" in name:
            param.requires_grad = True

        if bias == "all":
            if "bias" in name:
                param.requires_grad = True
        elif bias == "lora_only":
            if "lora_" in name and "bias" in name:
                param.requires_grad = True
        elif bias == "ia3_only":
            if "ia3_" in name and "bias" in name:
                param.requires_grad = True


def pbllm_mark_adapters_as_trainable(model: nn.Module, bias: str = "none"):
    # Make adapter-specific parameters require gradient
    for name, param in model.named_parameters():
        # param.requires_grad = False (testing for grad_norm temporariliy)

        if (
            "lora_" in name
            or "ia3_" in name
            or "vera_" in name
            or "loha_" in name
            or "existing_" in name
        ):  # or "classification" in name:
            param.requires_grad = True
        if bias == "all":
            if "bias" in name:
                param.requires_grad = True
        elif bias == "lora_only":
            if "lora_" in name and "bias" in name:
                param.requires_grad = True
        elif bias == "ia3_only":
            if "ia3_" in name and "bias" in name:
                param.requires_grad = True

    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()


def collect_all_peft_params(model):
    """
    Collects all parameters and their associated learning rates from all PEFTAdapter modules in the model.
    """
    all_params = []
    for module in model.modules():
        if isinstance(module, PEFTAdapter):
            all_params.extend(module.get_params_lr())
    return all_params
