# pylint: disable=C0413
import os

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../../"))

import gc
import inspect
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from sscompiler.compiler.data_utils import get_loader_from_dataset

from .data_utils import prepare_calibration_input
from .layers.fp4 import Portable4BitLinear
from .layers.loftq import PortableLoftQLayer

# from .layers.pbllm2 import GPTQLayer
from .layers.wanda import WandaLayer


class AbstractTransformer:
    def __init__(
        self,
        model_dir: str,
        groups: Dict,
        auto_model=None,
        eval_model=None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        self.model_dir = model_dir

        if auto_model is None:
            if "t5" in model_dir:
                self.auto_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_dir,
                    torch_dtype=dtype,
                    device_map=device_map,
                )
            else:
                self.auto_model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=dtype,
                    device_map=device_map,
                )
        else:
            self.auto_model = auto_model

        self.eval_model = eval_model
        self.dtype = dtype
        self.groups = groups

    def __repr__(self) -> str:
        return self.auto_model.__repr__()

    def match_submodules(
        self, key: str, indexable_module: str = None, idxs: List[int] = None
    ) -> List[str]:
        modules = [
            module
            for module, _ in self.auto_model.named_modules()
            if module.endswith(f".{key}")
        ]
        # TODO: fix assume same structure
        if indexable_module is not None and idxs is not None:
            int_idx = modules[0].split(".").index(indexable_module) + 1
            modules = [i for i in modules if int(i.split(".")[int_idx]) in idxs]
        return modules

    def get_submodule(self, module_name: str):
        return self.auto_model.get_submodule(module_name)

    def replace_submodule(self, module_path: str, new_module):
        parts = module_path.split(".")
        current_module = self.auto_model

        for part in parts[:-1]:
            current_module = current_module._modules.get(part)
            if current_module is None:
                raise ValueError(
                    f"Module path '{module_path}' is invalid. '{part}' does not exist."
                )

        if parts[-1] in current_module._modules:
            del current_module._modules[parts[-1]]
            gc.collect()
            torch.cuda.empty_cache()
            current_module._modules[parts[-1]] = new_module

        elif parts[-1] in current_module._parameters:
            del current_module._parameters[parts[-1]]
            gc.collect()
            torch.cuda.empty_cache()
            current_module._parameters[parts[-1]] = new_module
        else:
            raise ValueError(
                f"Submodule '{parts[-1]}' not found in path '{module_path}'."
            )

    def get_parameter(self, param_name: str):
        return self.auto_model.get_parameter(param_name)

    # add allowed types as parameter
    # mapping between optimization type and suppported layers - constants.py file in layers folder
    def inject_adapter(
        self,
        replacement_groups: List[str],
        adapter_fn,
        indexable_module: str = None,
        idxs: List[int] = None,
    ):
        """
        Add an adapter to the model for the specified groups
        """
        for group in replacement_groups:
            assert group in self.groups
            for module_path in self.match_submodules(
                self.groups[group], indexable_module=indexable_module, idxs=idxs
            ):
                original_module = self.get_submodule(module_path)
                original_device = next(original_module.parameters()).device
                new_module = adapter_fn(original_module)
                new_module = new_module.to(original_device)
                self.replace_submodule(module_path, new_module)

    def get_tokenizer(
        self,
        add_eos_token=False,
        add_bos_token=False,
        pad_token=None,
        padding_side=None,
    ):
        """
        Gets the tokenizer for the model
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = add_eos_token
        tokenizer.add_bos_token = add_bos_token
        if pad_token:
            if pad_token == "eos":
                pad_token = tokenizer.eos_token
            tokenizer.pad_token = pad_token
            tokenizer.padding_side = padding_side
        else:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @torch.no_grad
    def quant(self, dataset, nsamples=128):
        """
        Quantize
        """

        print("Loading data")
        tokenizer = self.get_tokenizer(True, False)

        seqlen = 1024

        calibration_data = get_loader_from_dataset(
            dataset=dataset, nsamples=nsamples, seqlen=seqlen, tokenizer=tokenizer
        )

        print("Done loading")
        model = self.auto_model

        # print("getting input, output, attention mask and position ids")
        dev = torch.device(self.auto_model.device)
        with torch.no_grad():
            # load dataset on each device;
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model,
                calibration_data,
                dev,
                is_decoder_only=True,
            )

        if "OPT" in model.__class__.__name__:
            layers = model.model.decoder.layers
        else:
            layers = model.model.layers
        for layer in tqdm(layers, desc="Quantizing model"):
            subset = self.find_layers(
                layer,
                layers=[GPTQLayer],
            )
            dev = torch.device("cuda:0")

            if "OPT" not in model.__class__.__name__:
                inps, outs, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    position_ids.to(dev),
                )
            else:
                inps, outs, position_ids = inps.to(dev), outs.to(dev), None

            def add_batch(gptq_layer):
                def tmp(_, inp, out):
                    gptq_layer.add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for sublayer in subset.values():
                if isinstance(sublayer, GPTQLayer):
                    handles.append(sublayer.register_forward_hook(add_batch(sublayer)))
                    sublayer.createH()

            for j in range(nsamples):
                with torch.no_grad():
                    if "OPT" in model.__class__.__name__:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
            for h in handles:
                h.remove()
            for sublayer in subset.values():
                if isinstance(sublayer, GPTQLayer):
                    sublayer.quantize()
                    sublayer.free()
            for j in range(nsamples):
                with torch.no_grad():
                    if "OPT" in model.__class__.__name__:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
            del subset
            del layer
            inps, outs = outs, inps
            torch.cuda.empty_cache()

    def prune(
        self,
        dataset,
        nsamples=128,
        is_decoder_only=True,
        seqlen=1024,
    ):
        """
        Prunes the layer by passing data through the layers of the model, storing key information in this forward pass through the handles, and using that data to prune
        Most of this work is done within the Pruning layer itself (such as WandaLayer). Pruning layer needs to have 'add_batch' and 'prune' functions which handle
        which data to store during forward pass and how to use that to prune weights
        """

        print("Loading data")
        tokenizer = self.get_tokenizer(True, False)

        calibration_data = get_loader_from_dataset(
            dataset=dataset, nsamples=nsamples, seqlen=seqlen, tokenizer=tokenizer
        )

        print("Done loading")
        model = self.auto_model
        dev = torch.device("cuda:0")

        with torch.no_grad():
            if is_decoder_only:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model,
                    calibration_data,
                    dev,
                    seqlen=seqlen,
                    is_decoder_only=is_decoder_only,
                )
            else:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model,
                    calibration_data,
                    dev,
                    seqlen=seqlen,
                    is_decoder_only=is_decoder_only,
                )

        def prune_layers(layers, inps, outs, attention_mask, position_ids):
            for _, layer in enumerate(layers):
                print("Pruned a layer")
                subset = self.find_layers(
                    layer,
                    layers=[
                        nn.Linear,
                        WandaLayer,
                        Portable4BitLinear,
                        PortableLoftQLayer,
                    ],
                )
                dev = torch.device("cuda:0")
                if position_ids is not None:
                    inps, outs, position_ids = (
                        inps.to(dev),
                        outs.to(dev),
                        position_ids.to(dev),
                    )
                else:
                    (
                        inps,
                        outs,
                    ) = inps.to(
                        dev
                    ), outs.to(dev)

                def add_batch(prune_layer):
                    def tmp(_, inp, out):
                        prune_layer.add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for _, sublayer in subset.items():
                    if isinstance(sublayer, WandaLayer):
                        handles.append(
                            sublayer.register_forward_hook(add_batch(sublayer))
                        )
                for j in range(nsamples):
                    with torch.no_grad():
                        if position_ids is not None:
                            outs[j] = layer(
                                inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                            )[0]
                        else:
                            outs[j] = layer(
                                inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                            )[0]
                for h in handles:
                    h.remove()
                for name, sublayer in subset.items():
                    if isinstance(sublayer, WandaLayer):
                        subset[name].prune()
                for j in range(nsamples):
                    with torch.no_grad():
                        if position_ids is not None:
                            outs[j] = layer(
                                inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                            )[0]
                        else:
                            outs[j] = layer(
                                inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                            )[0]
                inps, outs = outs, inps

                # replace all wanda layers with their OG linear layers
                for name, sublayer in subset.items():
                    for mod_name, mod in model.named_modules():
                        if mod is sublayer and isinstance(mod, WandaLayer):
                            self.replace_submodule(mod_name, sublayer.get_layer())

        if is_decoder_only:
            # Prune decoder-only model layers
            layers = model.model.layers
            prune_layers(layers, inps, outs, attention_mask, position_ids)
        else:
            # Prune encoder-decoder model layers
            encoder_layers = model.transformer.encoder.block
            decoder_layers = model.transformer.decoder.block

            prune_layers(encoder_layers, inps, outs, attention_mask, position_ids)
            prune_layers(decoder_layers, inps, outs, attention_mask, position_ids)

    def get_activations(self, dataset, nsamples=128, is_decoder_only=True, seqlen=1024):
        """
        Prunes the layer by passing data through the layers of the model, storing key information in this forward pass through the handles, and using that data to prune
        Most of this work is done within the Pruning layer itself (such as WandaLayer). Pruning layer needs to have 'add_batch' and 'prune' functions which handle
        which data to store during forward pass and how to use that to prune weights
        """

        activation_dict = {}
        print("Loading data")
        tokenizer = self.get_tokenizer(True, False)

        calibration_data = get_loader_from_dataset(
            dataset=dataset, nsamples=nsamples, seqlen=seqlen, tokenizer=tokenizer
        )

        print("Done loading")
        model = self.auto_model
        dev = torch.device("cuda:0")

        with torch.no_grad():
            if is_decoder_only:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model,
                    calibration_data,
                    dev,
                    seqlen=seqlen,
                    is_decoder_only=is_decoder_only,
                )
            else:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(
                    model,
                    calibration_data,
                    dev,
                    seqlen=seqlen,
                    is_decoder_only=is_decoder_only,
                )

        layers = model.model.layers
        for layer_idx, layer in enumerate(layers):
            dev = torch.device("cuda:0")
            if position_ids is not None:
                inps, outs, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    position_ids.to(dev),
                )
            else:
                (
                    inps,
                    outs,
                ) = inps.to(
                    dev
                ), outs.to(dev)

            activations_before = []
            for j in range(nsamples):
                with torch.no_grad():
                    if position_ids is not None:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
                    activations_before.append(outs[j].cpu().to(torch.float16).numpy())

            activation_dict[f"layer_{layer_idx + 1}"] = np.array(activations_before)

        return activation_dict

    def check_sparsity(self):
        """
        Checks the sparsity ratio of the different layers in the model's weights
        """
        layers = self.auto_model.model.layers
        count = 0
        total_params = 0
        for i, layer in layers:
            subset = self.find_layers(
                layer, layers=[nn.Linear, WandaLayer, Portable4BitLinear]
            )

            sub_count = 0
            sub_params = 0
            for name, sublayer in subset.items():
                W = sublayer.weight.data
                count += (W == 0).sum().item()
                total_params += W.numel()

                sub_count += (W == 0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        return float(count) / total_params

    def find_layers(self, module, layers=[nn.Linear], name=""):
        """
        Recursively find the layers of a certain type in a module.

        Args:
            module (nn.Module): PyTorch module.
            layers (list): List of layer types to find.
            name (str): Name of the module.

        Returns:
            dict: Dictionary of layers of the given type(s) within the module.
        """

        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                self.find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.auto_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.auto_model.parameters() if p.requires_grad
        )
        total_memory_bytes = sum(
            p.numel() * p.element_size() for p in self.auto_model.parameters()
        )
        total_memory_mb = total_memory_bytes / (1024**2)

        print(f"Total Memory Size: {total_memory_mb:.2f} MB")
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

        return total_memory_mb, total_params, trainable_params

    def cleanup_memory(self) -> None:
        """Run GC and clear GPU memory."""
        caller_name = ""
        try:
            caller_name = f" (from {inspect.stack()[1].function})"
        except (ValueError, KeyError):
            pass

        def total_reserved_mem() -> int:
            return sum(
                torch.cuda.memory_reserved(device=i)
                for i in range(torch.cuda.device_count())
            )

        memory_before = total_reserved_mem()

        # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after = total_reserved_mem()
            print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

    def prep_for_save(self):
        sparse_params = []
        for name, param in self.auto_model.named_parameters():
            if getattr(param, "is_sparse_csr", False):
                sparse_params.append(name)

        for param_name in sparse_params:
            sparse_param = self.get_parameter(param_name)
            dense_param = sparse_param.to_dense()
            self.replace_submodule(param_name, dense_param)

    def sparsify(self, suffix):
        dense_params = []
        for name, _ in self.auto_model.named_parameters():
            if name.endswith(suffix):
                dense_params.append(name)

        for param_name in dense_params:
            dense_param = self.get_parameter(param_name)
            sparse_param = dense_param.to_sparse_csr()
            self.replace_submodule(param_name, dense_param)
