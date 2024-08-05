"""
Script for running a single experiment with a specified baseline using a fixed learning rate and r value.
"""

# pylint: disable=C0413
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import gc
import logging

import h5py
import jsbeautifier
import setproctitle
import torch
from datasets import load_dataset
from sscompiler.compiler import AbstractTransformer
from sscompiler.compiler.layers import (
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
)
from sscompiler.utils.argument_classes import (
    ExperimentOptions,
    SearchOptions,
    SlimscaleParser,
)
from sscompiler.utils.constants import METRIC_MAP, SUPERGLUE_DATASETS, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, glue_activations, nf4, wanda
from sscompiler.utils.tokenization import tokenize_glue
from transformers import AutoConfig, AutoModelForSequenceClassification

options = jsbeautifier.default_options()
options.indent_size = 2

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests")
STORAGE_DIR = "/srv/shared_home/common-data/slimscale/slimscale-databases"


def ia3(at, r):
    at.inject_adapter(
        ["value", "key"],
        lambda x: PortableIA3Adapter(
            x,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )
    at.inject_adapter(
        ["dense2"],
        lambda x: PortableIA3Adapter(
            x,
            in_features=x.in_features,
            out_features=x.out_features,
            is_feedforward=True,
        ),
    )


def lora(at, r):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoRAAdapter(
            x,
            r=r,
            bias=False,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def loha(at, r):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoHAAdapter(
            x,
            r=r,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def vera(at, r):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableVeRAAdapter(
            x,
            r=r,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


OPT_MAP = {
    "ia3": ia3,
    "lora": lora,
    "loha": loha,
    "vera": vera,
}


def create_experiment(args):
    """
    Runs a single experiment with a fixed learning rate and r value.
    """
    task = args.task
    is_regression = args.task == "stsb"
    if not is_regression:
        raw_datasets = load_dataset(
            "super_glue" if args.task in SUPERGLUE_DATASETS else "glue",
            args.task,
        )
        num_labels = len(raw_datasets["train"].features["label"].names)
        del raw_datasets
        gc.collect()
    else:
        num_labels = 1

    if "t5" in args.model or ("gemma" in args.model and task == "stsb"):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    target_modules = TARGET_MODULES[args.model]
    auto_config = AutoConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task=task,
    )

    auto_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        config=auto_config,
        ignore_mismatched_sizes=True,
    )
    at = AbstractTransformer(
        model_dir=args.model,
        groups=target_modules,
        auto_model=auto_model,
    )

    if task in SUPERGLUE_DATASETS:
        raw_datasets = load_dataset("super_glue", task)
    else:
        raw_datasets = load_dataset(
            "glue",
            task,
        )
    activations = glue_activations(at, raw_datasets["train"], task)

    def save_activations_to_hdf5(activation_dict, file_path):
        with h5py.File(file_path, "w") as f:
            for key, value in activation_dict.items():
                f.create_dataset(key, data=value)

    save_activations_to_hdf5(activations, "gemma_cola_activations.h5")


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Run a single experiment with a fixed configuration on a given task"
    )

    parser.add_arguments(SearchOptions, dest="validation")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    cli = parser.parse_args()

    setproctitle.setproctitle(
        f"Slimscale Activations ({cli.task}) -O{cli.num_opts}"
        + (" DEBUG" if cli.debug_mode else "")
    )

    if cli.should_pad and "t5" in cli.model:
        raise RuntimeError("Should not pad T5.")

    model_name = cli.model.split("/")[-1]
    log_dir = os.path.join(BASE_DIR, "logs", "activations")
    log_name = (
        f"epochs[{cli.epochs}]"
        f"_batch_size[{cli.batch_size}]"
        f"_max_length_[{cli.max_length}]"
        f"_train_head_[{cli.train_head}]"
        f"_should_pad_[{cli.should_pad}]"
        f"_quantized_[{cli.quantize}]"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.out")
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("\n--------------------------------")
    logger.info("Using devices: %s", os.environ["CUDA_VISIBLE_DEVICES"])
    logger.info("Using seed: %d", seed)

    logger.info("Start single experiment.")
    create_experiment(cli)
