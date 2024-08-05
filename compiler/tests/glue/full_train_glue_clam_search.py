# pylint: disable=C0413
"""
Searches a SQLite database of optuna trials for the top performing
configurations on a task, and then trains a model with these configurations to
completion
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

global_hyperparams = {
    "loha": {"learning_rate": 0.001},
    "vera": {"learning_rate": 0.1},
    "ia3": {"learning_rate": 0.01},
    "lora": {"learning_rate": 0.0004},
}
logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests")


def ia3(at, scale_factor, **kwargs):
    ia3_lr = global_hyperparams["ia3"]["learning_rate"] * scale_factor
    at.inject_adapter(
        ["value", "key"],
        lambda x: PortableIA3Adapter(
            x,
            in_features=x.in_features,
            out_features=x.out_features,
            ia3_lr=ia3_lr,
        ),
    )
    at.inject_adapter(
        ["dense2"],
        lambda x: PortableIA3Adapter(
            x,
            in_features=x.in_features,
            out_features=x.out_features,
            is_feedforward=True,
            ia3_lr=ia3_lr,
        ),
    )


def lora(at, scale_factor, lora_r, **kwargs):
    lora_lr = global_hyperparams["lora"]["learning_rate"] * scale_factor
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoRAAdapter(
            x,
            r=lora_r,
            bias=False,
            in_features=x.in_features,
            out_features=x.out_features,
            lora_lr=lora_lr,
        ),
    )


def loha(at, scale_factor, loha_r, **kwargs):
    loha_lr = global_hyperparams["loha"]["learning_rate"] * scale_factor
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoHAAdapter(
            x,
            r=loha_r,
            in_features=x.in_features,
            out_features=x.out_features,
            loha_lr=loha_lr,
        ),
    )


def vera(at, scale_factor, vera_r, **kwargs):
    vera_lr = global_hyperparams["vera"]["learning_rate"] * scale_factor
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableVeRAAdapter(
            x,
            r=vera_r,
            in_features=x.in_features,
            out_features=x.out_features,
            vera_lr=vera_lr,
        ),
    )


OPT_MAP = {
    "ia3": ia3,
    "lora": lora,
    "loha": loha,
    "vera": vera,
}


def main(args):
    trials = get_hyperparams(args.database)

    task = args.task

    is_regression = task == "stsb"
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
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        config=auto_config,
        ignore_mismatched_sizes=True,
    )
    target_modules = TARGET_MODULES[args.model]

    auto_model.config.pad_token_id = auto_model.config.eos_token_id

    at = AbstractTransformer(
        model_dir=args.model,
        groups=target_modules,
        auto_model=auto_model,
        device_map="auto",
    )

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
        task=args.task,
        model=at.auto_model,
        should_pad=args.should_pad,
        max_length=args.max_length,
        full_train=True,
    )

    target_modules = TARGET_MODULES[args.model]
    for i, trial in trials.items():
        logger.info("--------------------------------")
        logger.info("Trial %d", i)
        at = AbstractTransformer(
            model_dir=args.model,
            groups=target_modules,
            auto_model=auto_model,
            device_map="auto",
        )

        if args.prune:
            wanda(at, raw_datasets["train"], args.task)
        if args.quantize:
            nf4(at)

        learning_rate = trial.pop("learning_rate", 1e-4)
        logger.info("Trial %d Learning Rate: %f", i, learning_rate)
        opts = trial["opts"]
        scale_factor = trial["scale_factor"]
        logger.info("Scale factor is: %f", scale_factor)

        for j, o in enumerate(opts):
            opt = o.pop("opt")
            logger.info(
                "Opt %d: Applying %s to model with hyperparameters: %s", j + 1, opt, o
            )
            OPT_MAP[opt](at, scale_factor=scale_factor, **o)

        result = finetune_at(
            at=at,
            task=args.task,
            tokenized_train=tokenized_train,
            tokenized_eval=tokenized_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            train_head=args.train_head,
            use_multi_lr=True,
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
    args = parser.parse_args()
    model = args.model
    task = args.task
    model_name = model.split("/")[-1]
    num_opts = re.search(r"num_opts_\[(\d+)\]", args.database).group(1)

    log_dir = os.path.join(
        TEST_DIR,
        "logs",
        "validation",
        "part-b",
        model_name,
        task,
        f"{num_opts}-opts",
    )
    log_name = (
        f"epochs[{args.epochs}]"
        f"_batch_size[{args.batch_size}]"
        f"_max_length_[{args.max_length}]"
        f"_train_head_[{args.train_head}]"
        f"_should_pad_[{args.should_pad}]"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.out")
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("\n--------------------------------")
    logger.info("Using devices: %s", os.environ["CUDA_VISIBLE_DEVICES"])
    setproctitle.setproctitle(f"Slimscale, ({task}), full ft, order {num_opts}")

    all_scores = []
    for seed in [0, 10, 100, 1000, 10000]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.info("Using seed: %d", seed)

        metrics, trainable_params = main(args)
        all_scores.append(metrics)
        logger.info("seed %d finished with result %f", seed, metrics)

    avg = np.average(all_scores)
    std = np.std(all_scores)

    logger.info(all_scores)
    logger.info("average: %f", avg)
    logger.info("stdev: %f", std)
    logger.info("trainable_params: %d", trainable_params)
