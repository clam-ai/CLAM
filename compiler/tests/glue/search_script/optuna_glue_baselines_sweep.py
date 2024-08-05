"""
Script for finding the optimal hyperparameters of an optimization technique
using the `Optuna` library
"""

# pylint: disable=C0413
import os

os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import gc
import logging

import jsbeautifier
import optuna
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
from sscompiler.utils.experiments import finetune_at, nf4, wanda
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


def ia3(trial, at, opt_num, r):
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


def lora(trial, at, opt_num, r):
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


def loha(trial, at, opt_num, r):
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoHAAdapter(
            x,
            r=r,
            in_features=x.in_features,
            out_features=x.out_features,
        ),
    )


def vera(trial, at, opt_num, r):
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


def create_objective(args):
    """
    Returns an optuna objective function for a given model configuration.

        config: {
            model:          model name
            target_modules: target module map
            pre:            ordered list of pre-finetune optimizations*
            post:           ordered list of post-finetune optimizations*
        }
            * be sure to add optimizations and their appropriate optimizable
              methods to OPT_MAP (see lora and quantize for examples)

        task:   name of the task to optimize over; be sure it is included in
                DATASET_MAP and METRIC_MAP so that finetune_at and eval_at work

        search_over_opts:   False if you want to optuna to perform hyperparameter
                            search for the optimization configuration given in config;
                            True if you want optuna to ignore the optimizations in config
                            and search over a pre-defined space of optimizations
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

    def objective(trial):
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

        if args.prune:
            wanda(at, raw_datasets["train"], task)
        if args.quantize:
            nf4(at)

        gc.collect()
        torch.cuda.empty_cache()

        learning_rate = trial.suggest_categorical(
            "learning_rate",
            [1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3, 1e-2, 2e-2, 4e-2, 1e-1, 2e-1, 4e-1],
        )
        opt_num = 1
        if args.baseline == "vera":
            r = trial.suggest_categorical(f"r_{opt_num}", [64, 128, 256, 512, 1024])
        elif args.baseline == "ia3":
            r = 0
        else:
            r = trial.suggest_categorical(f"r_{opt_num}", [4, 8, 16, 32, 64])
        OPT_MAP[args.baseline](trial, at, opt_num, r)

        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info("Trial: %d", trial.number)
        logger.info("Parameters: %s", trial.params)
        logger.info("Total Memory Size: %.2f MB", total_memory_mb)
        logger.info("Total Parameters: %d", total_params)
        logger.info("Trainable Parameters: %d", trainable_params)

        if "gemma" in args.model:
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
            full_train=False,
        )

        result = finetune_at(
            at=at,
            task=args.task,
            tokenized_train=tokenized_train,
            tokenized_eval=tokenized_test,
            epochs=args.epochs,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            train_head=args.train_head,
            use_multi_lr=False,
            tokenizer=tokenizer,
        )

        if isinstance(result, tuple):
            return result[0][0]
        else:
            return result

    return objective


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline to brute-force",
    )
    parser.add_arguments(SearchOptions, dest="validation")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    cli = parser.parse_args()

    setproctitle.setproctitle(
        f"Slimscale Baseline {cli.baseline} ({cli.task}) -O{cli.num_opts}"
        + (" DEBUG" if cli.debug_mode else "")
    )

    if cli.should_pad and "t5" in cli.model:
        raise RuntimeError("Should not pad T5.")

    model_name = cli.model.split("/")[-1]
    study_name = (
        f"sweep_lr_model_[{model_name}]"
        f"_task_[{cli.task}]"
        f"_epochs[{cli.epochs}]"
        f"_opt_[{cli.baseline}]"
        f"_batch_size[{cli.batch_size}]"
        f"_max_length_[{cli.max_length}]"
        f"_train_head_[{cli.train_head}]"
        f"_should_pad_[{cli.should_pad}]"
        f"_quantized_[{cli.quantize}]"
    )
    if cli.debug_mode:
        study_name = "debug"

    log_dir = os.path.join(
        BASE_DIR,
        "logs",
        model_name,
        "experiment-1" if cli.quantize is False else "experiment-2",
        "baselines",
        cli.task,
        cli.baseline,
    )
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

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    db_dir = os.path.join(
        BASE_DIR,
        model_name,
        "experiment-1" if cli.quantize is False else "experiment-2",
        "baselines",
        cli.task,
    )
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{study_name}.db")
    SQLITE_DB = f"sqlite:///{db_path}"

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
        direction="maximize",
        study_name=study_name,
        storage=SQLITE_DB,
        load_if_exists=True,
    )

    logger.info("Start optimization.")
    study.optimize(
        create_objective(
            cli,
        ),
        n_trials=cli.trials,
        gc_after_trial=True,
        catch=[torch.cuda.OutOfMemoryError],
    )
