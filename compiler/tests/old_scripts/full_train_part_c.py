"""
Searches a SQLite database of optuna trials for the top performing
configurations on a task, and then trains a model with these configurations to
completion
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sscompiler.utils.constants import (
    SUPERGLUE_DATASETS,
    TARGET_MODULES,
)
from sscompiler.compiler import (
    AbstractTransformer,
)
from sscompiler.compiler.layers import (
    PortableLoRAAdapter,
    PortableIA3Adapter,
    PortableVeRAAdapter,
    PortableLoHAAdapter,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    T5ForSequenceClassification,
)
from datasets import load_dataset
import setproctitle
import torch
import time
import logging
import argparse
import os
from sscompiler.utils.experiments import finetune_at, get_hyperparams, wanda, nf4


global_hyperparams = {
    "lora": {"learning_rate": 0.0004},
    "vera": {"learning_rate": 0.1},
    "ia3": {"learning_rate": 0.01},
    "loha": {"learning_rate": 0.004},
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TASK_DIR = os.path.join(BASE_DIR, "tasks")
LOG_FILE = os.path.join(TASK_DIR, ".tmp/full_train.log")
logging.basicConfig(filename=LOG_FILE, encoding="utf8", level=logging.INFO)


def ia3(at, **kwargs):
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


def lora(at, **kwargs):
    lora_r = kwargs.pop("lora_r")
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoRAAdapter(
            x,
            r=lora_r,
            bias=False,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def loha(at, **kwargs):
    loha_r = kwargs.pop("loha_r")
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoHAAdapter(
            x,
            r=loha_r,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def vera(at, **kwargs):
    vera_r = kwargs.pop("vera_r")
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableVeRAAdapter(
            x,
            r=vera_r,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def bitfit(at):
    at.inject_adapter(
        ["query", "gate"],
        # BitFit
    )


OPT_MAP = {
    "ia3": ia3,
    "lora": lora,
    "loha": loha,
    "vera": vera,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_argument(
        "--task", type=str, help="The name of the GLUE task", default="cola"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="/home/ubuntu/slimscale/compiler/tests/databases/part_c_results_2_opt/part_c_model_[t5-base]_task_[cb]_trial_[100]_epochs[5]_num_opts_[2]_batch_size[32]_max_length_[512]_train_head_[False]_should_pad_[False]_quantized_[False].db",
        help="Absolute path to the SQLite database to query",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="The name of the base model to train",
        default="google-t5/t5-base",
    )
    parser.add_argument("--seed", type=int, help="The torch / random seed", default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    # should be no for T5, yes for experiment 2
    parser.add_argument("--train_head", type=bool, default=True)
    # should be no for T5, yes for gemma
    parser.add_argument("--should_pad", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--prune", type=str, default=False)
    parser.add_argument("--sparsity_ratio", type=int, default=0.5)
    parser.add_argument("--structured", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    setproctitle.setproctitle(
        f"Slimscale, ({args.task}), full ft, order 2, Top 1 Model"
    )
    trials = get_hyperparams(args.database)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Using seed: {seed}")

    raw_datasets = load_dataset(
        "super_glue" if args.task in SUPERGLUE_DATASETS else "glue",
        args.task,
    )
    is_regression = args.task == "stsb"
    if not is_regression:
        num_labels = len(raw_datasets["train"].features["label"].names)
    else:
        num_labels = 1

    auto_config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        finetuning_task=args.task,
    )
    if "t5" in args.base_model:
        auto_model = T5ForSequenceClassification.from_pretrained(
            args.base_model,
            device_map="cuda",
            config=auto_config,
            ignore_mismatched_sizes=True,
        )
    elif "gemma" in args.base_model and args.task == "stsb":
        auto_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            device_map="cuda",
            config=auto_config,
            ignore_mismatched_sizes=True,
        )
    else:
        auto_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            config=auto_config,
            ignore_mismatched_sizes=True,
        )

    target_modules = TARGET_MODULES[args.base_model]
    for i, trial in trials.items():
        logger.info("--------------------------------")
        logger.info("Trial %d", i)
        at = AbstractTransformer(
            model_dir=args.base_model,
            groups=target_modules,
            auto_model=auto_model,
            device_map="cuda",
        )
        if args.prune:
            wanda(at, raw_datasets["train"], args.task, **args.__dict__)
        if args.quantize:
            nf4(at)
        learning_rate = trial.pop("learning_rate", 1e-4)
        print("Trial" + str(i) + " has Learning Rate: " + str(learning_rate))
        opts = trial["opts"]

        start = time.time()
        for j, o in enumerate(opts):
            opt = o.pop("opt")
            logger.info(
                "Opt %d: Applying %s to model with hyperparameters: %s", j + 1, opt, o
            )

            print(
                "Opt "
                + str(j + 1)
                + ": Applying "
                + opt
                + " to model with hyperparameters: "
            )
            print(o)
            OPT_MAP[opt](at, **o)
        end = time.time()
        logger.info(f"Optimization time: {end - start}")

        task = args.pop("task")
        metrics = finetune_at(
            at, task, learning_rate=learning_rate, full_train=True, **args.__dict__
        )
        print("BEST ACCURACY:", metrics)
        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )
        logger.info(f"Total Memory Size: {total_memory_mb:.2f} MB")
        logger.info(f"Total Parameters: {total_params}")
        logger.info(f"Trainable Parameters: {trainable_params}")


if __name__ == "__main__":
    args = parse_args()
    logger.addHandler(logging.FileHandler(LOG_FILE))
    logger.info("\n--------------------------------")
    logger.info(f"Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    main()
