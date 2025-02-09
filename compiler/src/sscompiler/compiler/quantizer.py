from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import PreTrainedModel
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name

from .layers.quantized import QuantizedLayer


class SSQuantizer(HfQuantizer):
    requires_parameters_quantization = True
    requires_calibration = False

    def __init__(
        self,
        quantization_config,
        parameter_class=None,
        **kwargs,
    ):
        super().__init__(quantization_config, **kwargs)
        self.target_dtype: "torch.dtype" = kwargs.get("target_dtype")

        # this should actually come from the quantization config
        self.target_class: QuantizedLayer = kwargs.get("target_class")
        self.param_class = parameter_class
        self._is_trainable = kwargs.get("is_trainable")
        self._is_serializable = kwargs.get("is_serializable")

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        device_map = kwargs.get("device_map", None)
        if device_map is not None and isinstance(device_map, dict):
            device_map_without_lm_head = {
                key: device_map[key]
                for key in device_map.keys()
                if key not in self.modules_to_not_convert
            }
            if (
                "cpu" in device_map_without_lm_head.values()
                or "disk" in device_map_without_lm_head.values()
            ):
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        return self.target_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(
            module._parameters.get(tensor_name, None),
            self.param_class,
        ):
            return True
        elif isinstance(module, self.target_class) and tensor_name == "bias":
            # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
            # but it would wrongly use uninitialized weight there.
            return True
        else:
            return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(
                f"{module} does not have a parameter or a buffer named {tensor_name}."
            )

        old_value = getattr(module, tensor_name)

        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(
                new_value, requires_grad=old_value.requires_grad
            )
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], self.param_class):
            raise ValueError("this function only loads Portable4BitLinear components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(
                f"{tensor_name} is on the meta device, we need a `value` to put it on {target_device}."
            )

        if self.pre_quantized:
            if not self.is_serializable:
                raise ValueError(
                    "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                    "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                )

            if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possible other `quantized_stats` components."
                )

            quantized_stats = {}
            for k, v in state_dict.items():
                if param_name + "." in k:
                    quantized_stats[k] = v
                    if unexpected_keys is not None and k in unexpected_keys:
                        unexpected_keys.remove(k)

            new_value = self.param_class.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
            )
        else:
            new_value = param_value.to("cpu")

            kwargs = old_value.__dict__
            new_value = self.param_class(
                new_value,
                requires_grad=False,
                **kwargs,
            ).to(target_device)

        module._parameters[tensor_name] = new_value

    def adjust_max_memory(
        self, max_memory: Dict[str, Union[int, str]]
    ) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            print(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
            print(
                "The device_map was not initialized. "
                "Setting device_map to {'':torch.cuda.current_device()}. "
                "If you want to use the model for inference, please set device_map = 'auto' "
            )
        return device_map

    def _process_model_before_weight_loading(
        self,
        model,
        device_map,
        keep_in_fp32_modules: List[str],
        **kwargs,
    ):
        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [
                key for key, value in device_map.items() if value in ["disk", "cpu"]
            ]

            if len(keys_on_cpu) > 0:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_portable_quantized(
            model,
            replacement_layer=self.target_class,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )
        model.config.quantization_config = self.quantization_config
        # torch dtypes are not serializable apparently, so we need to convert it to a string first
        model.config.quantization_config.original_dtype = str(
            model.config.quantization_config.original_dtype
        )
        model.config._hf_clam_added = True

    def _process_model_after_weight_loading(self, model, **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable
        return model

    @property
    def is_trainable(self):
        return self._is_trainable

    @property
    def is_serializable(self):
        return self._is_serializable


def replace_with_portable_quantized(
    model: torch.nn.Module,
    replacement_layer: QuantizedLayer,
    modules_to_not_convert: List[str] = None,
    current_key_name: str = None,
    quantization_config=None,
):
    model, has_been_replaced = _replace_with_portable_quantized(
        model,
        replacement_layer=replacement_layer,
        modules_to_not_convert=modules_to_not_convert,
        current_key_name=current_key_name,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        print(
            "You are loading your quantized model but no quantized modules were found in your model."
        )

    return model


def _replace_with_portable_quantized(
    model: torch.nn.Module,
    replacement_layer: QuantizedLayer,
    modules_to_not_convert: List[str],
    current_key_name: str = None,
    has_been_replaced=False,
    quantization_config=None,
) -> Tuple[torch.nn.Module, bool]:
    for name, module in model.named_children():
        current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str)
                for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = replacement_layer.empty_init(
                        in_features,
                        out_features,
                        module.bias is not None,
                        quantization_config=quantization_config,
                    )
                    has_been_replaced = True
                    model._modules[name].source_cls = replacement_layer
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_portable_quantized(
                model=module,
                replacement_layer=replacement_layer,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
                quantization_config=quantization_config,
            )

        current_key_name.pop(-1)

    return model, has_been_replaced
