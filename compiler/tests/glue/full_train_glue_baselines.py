# pylint: disable=C0413
"""
Script for generating baseline values on GLUE datasets
"""

import os

os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import logging
import re

import numpy as np
import setproctitle
import torch
from sscompiler.compiler import (
    AbstractTransformer,
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
)
from sscompiler.utils.argument_classes import (
    ExperimentOptions,
    SlimscaleParser,
    ValidationOptions,
)
from sscompiler.utils.constants import SUPERGLUE_DATASETS, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, get_hyperparams, nf4, wanda
from sscompiler.utils.tokenization import tokenize_glue
from transformers import AutoConfig, AutoModelForSequenceClassification

from datasets import load_dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def ia3(at, rank):
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


def lora(at, rank):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoRAAdapter(
            x,
            r=rank,
            bias=False,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def loha(at, rank):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoHAAdapter(
            x,
            r=rank,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def vera(at, rank):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableVeRAAdapter(
            x,
            r=rank,
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


def log_trainable_parameters(at: AbstractTransformer):
    total_params = sum(p.numel() for p in at.auto_model.parameters())
    trainable_params = sum(
        p.numel() for p in at.auto_model.parameters() if p.requires_grad
    )
    total_memory_bytes = sum(
        p.numel() * p.element_size() for p in at.auto_model.parameters()
    )
    total_memory_mb = total_memory_bytes / (1024**2)
    logger.info("Total Memory Size: %.2f MB", total_memory_mb)
    logger.info("Total Parameters: %d", total_params)
    logger.info("Trainable Parameters: %d", trainable_params)


def main(args):
    logger.info(args)
    experiment = args.experiment
    validation = args.validation

    task = args.task

    is_regression = args.task == "stsb"
    if task == "boolq":
        num_labels = 2
    elif not is_regression:
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
    config_kwargs = {}
    if task == "boolq" and "llama" in args.model:
        config_kwargs = {
            "problem_type": "single_label_classification",
        }

    auto_config = AutoConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task=task,
        **config_kwargs,
    )

    auto_model = AutoModelForSequenceClassification.from_pretrained(
        experiment.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        config=auto_config,
        ignore_mismatched_sizes=True,
    )
    target_modules = TARGET_MODULES[experiment.model]

    auto_model.config.pad_token_id = auto_model.config.eos_token_id

    at = AbstractTransformer(
        model_dir=args.model,
        groups=target_modules,
        auto_model=auto_model,
        device_map="auto",
    )

    if args.prune:
        wanda(at, raw_datasets["train"], experiment.task)
    if args.quantize:
        nf4(at)

    trials = get_hyperparams(validation.database)
    _, value = trials.popitem()
    rank = value.get("opts", [_, {"r": 0}])[1]["r"]
    OPT_MAP[baseline](at, rank)
    learning_rate = value["learning_rate"]

    if "gemma" in args.model or "llama" in args.model:
        tokenizer = at.get_tokenizer(
            add_bos_token=True,
            add_eos_token=True,
            pad_token="eos",
            padding_side="right",
        )
    else:
        tokenizer = at.get_tokenizer()

    tokenized_train, tokenized_test = tokenize_glue(
        tokenizer=tokenizer,
        task=experiment.task,
        model=at.auto_model,
        should_pad=experiment.should_pad,
        max_length=experiment.max_length,
        full_train=True,
    )

    result = finetune_at(
        at=at,
        task=experiment.task,
        tokenized_train=tokenized_train,
        tokenized_eval=tokenized_test,
        epochs=experiment.epochs,
        batch_size=experiment.batch_size,
        learning_rate=learning_rate,
        train_head=args.train_head,
        use_multi_lr=False,
        tokenizer=tokenizer,
    )

    logger.info(result)

    _, _, params = at.print_trainable_parameters()
    return result, params


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_arguments(ValidationOptions, dest="validation")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to run validation on",
        default=None,
    )
    cli = parser.parse_args()

    experiment = cli.experiment
    validation = cli.validation

    model_name = experiment.model.split("/")[-1]
    baseline = re.search(r"opt_\[(\w+)\]", validation.database).group(1)

    setproctitle.setproctitle(
        f"Slimscale baseline, ({model_name}), ({experiment.task}), ({baseline})"
    )

    log_dir = os.path.join(
        "logs",
        "validation",
        model_name,
        "experiment-1" if cli.quantize is False else "experiment-2",
        "baselines",
        experiment.task,
        baseline,
    )
    log_name = (
        f"epochs[{experiment.epochs}]"
        f"_batch_size[{experiment.batch_size}]"
        f"_max_length_[{experiment.max_length}]"
        f"_train_head_[{experiment.train_head}]"
        f"_should_pad_[{experiment.should_pad}]"
        f"_quantized_[{experiment.quantize}]"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.out")
    logger.addHandler(logging.FileHandler(log_file))

    all_scores = []

    if cli.seed is None:
        seeds = [0, 10, 100, 1000, 10000]
    else:
        seeds = [cli.seed]

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        metrics, trainable_params = main(cli)
        all_scores.append(metrics)
        logger.info("seed %d finished with result %f", seed, metrics)

        torch.cuda.empty_cache()
        gc.collect()

    avg = np.average(all_scores)
    std = np.std(all_scores)

    logger.info(all_scores)
    logger.info("average: %f", avg)
    logger.info("stdev: %f", std)
    logger.info("trainable_params: %d", trainable_params)
