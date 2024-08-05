"""
Script for finding the optimal hyperparameters of an optimization technique
using the `Optuna` library
"""

# pylint: disable=C0413
import os

os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import logging

import jsbeautifier
import optuna
import setproctitle
import torch
from sscompiler.compiler import (
    AbstractTransformer,
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
    mark_adapters_as_trainable,
)
from sscompiler.utils.argument_classes import (
    ExperimentOptions,
    SearchOptions,
    SlimscaleParser,
)
from sscompiler.utils.constants import SUPERGLUE_DATASETS, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, nf4, wanda
from sscompiler.utils.tokenization import tokenize_glue
from transformers import AutoConfig, AutoModelForSequenceClassification

from datasets import load_dataset

options = jsbeautifier.default_options()
options.indent_size = 2

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests")
STORAGE_DIR = "/srv/shared_home/common-data/slimscale/slimscale-databases"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

global_hyperparams = {
    "loha": {"learning_rate": 0.001},
    "vera": {"learning_rate": 0.1},
    "ia3": {"learning_rate": 0.01},
    "lora": {"learning_rate": 0.0004},
}


def ia3(trial, at, opt_num, scaling_factor):
    ia3_lr = global_hyperparams["ia3"]["learning_rate"] * scaling_factor
    at.inject_adapter(
        ["value", "key"],
        lambda x: PortableIA3Adapter(
            x, in_features=x.in_features, out_features=x.out_features, ia3_lr=ia3_lr
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


def lora(trial, at, opt_num, scaling_factor):
    lora_r = trial.suggest_categorical(f"lora_r_{opt_num}", [4, 8, 16, 32, 64])
    lora_lr = global_hyperparams["lora"]["learning_rate"] * scaling_factor
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


def loha(trial, at, opt_num, scaling_factor):
    loha_r = trial.suggest_categorical(f"loha_r_{opt_num}", [4, 8, 16, 32, 64])
    loha_lr = global_hyperparams["loha"]["learning_rate"] * scaling_factor
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


def vera(trial, at, opt_num, scaling_factor):
    vera_r = trial.suggest_categorical(f"vera_r_{opt_num}", [64, 128, 256, 512, 1024])

    vera_lr = global_hyperparams["vera"]["learning_rate"] * scaling_factor
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


def create_objective(
    args,
):
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
    num_opts = args.num_opts

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

    def objective(trial):
        auto_model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            config=auto_config,
            ignore_mismatched_sizes=True,
        )
        auto_model.config.pad_token_id = auto_model.config.eos_token_id
        at = AbstractTransformer(
            args.model,
            TARGET_MODULES[args.model],
            auto_model=auto_model,
        )

        if args.prune:
            wanda(at, raw_datasets["train"], task)
        if args.quantize:
            nf4(at)

        scale_factor = trial.suggest_categorical(
            "scale_factor",
            [0.1, 0.4, 1, 4],
        )

        opts = []
        for i in range(num_opts):
            opt_i = trial.suggest_categorical(f"opt_{i}", OPT_MAP.keys())
            opts.append(opt_i)
        for i, opt in enumerate(opts):
            OPT_MAP[opt](trial, at, i, scale_factor)

        mark_adapters_as_trainable(at.auto_model)
        if args.train_head:
            for name, param in at.auto_model.named_parameters():
                if "score" in name:
                    param.requires_grad = True

        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info("Trial: %d", trial.number)
        logger.info("Parameters: %s", trial.params)
        logger.info("Total Memory Size: %.2f MB", total_memory_mb)
        logger.info("Total Parameters: %d", total_params)
        logger.info("Trainable Parameters: %d", trainable_params)

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
            full_train=False,
        )

        result = finetune_at(
            at=at,
            task=args.task,
            tokenized_train=tokenized_train,
            tokenized_eval=tokenized_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_head=args.train_head,
            use_multi_lr=True,
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
    parser.add_arguments(SearchOptions, dest="validation")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    cli = parser.parse_args()

    setproctitle.setproctitle(
        f"Slimscale {cli.model} ({cli.task}) Part B Order {cli.num_opts}"
        + (" DEBUG" if cli.debug_mode else "")
    )

    if cli.should_pad and "t5" in cli.model:
        raise RuntimeError("Should not pad T5.")

    model_name = (cli.model).split("/")[-1]

    # name of the optuna study. should include all hyperparameters for easy querying / regex operations
    study_name = (
        f"model_[{model_name}]"
        f"_task_[{cli.task}]"
        f"_num_opts_[{cli.num_opts}]"
        f"_quantized_[{cli.quantize}]"
        f"_trial_[{cli.trials}]"
        f"_epochs[{cli.epochs}]"
        f"_batch_size[{cli.batch_size}]"
        f"_max_length_[{cli.max_length}]"
        f"_train_head_[{cli.train_head}]"
        f"_should_pad_[{cli.should_pad}]"
    )
    if cli.debug_mode:
        study_name = "debug"

    log_dir = os.path.join(
        TEST_DIR,
        "logs",
        "part-b",
        model_name,
        "experiment-1" if cli.quantize is False else "experiment-2",
        cli.task,
        f"{cli.num_opts}-opts",
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
    logger.info("Using hyperparameters: %s", global_hyperparams)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    direction = "minimize" if cli.task == "wikitext" else "maximize"

    db_dir = os.path.join(
        STORAGE_DIR,
        model_name,
        "experiment-1" if cli.quantize is False else "experiment-2",
        "part-b",
        f"{cli.num_opts}-opts",
    )
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{study_name}.db")
    SQLITE_DB = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=SQLITE_DB,
        load_if_exists=True,
    )

    logger.info("Start search.")
    study.optimize(
        create_objective(
            cli,
        ),
        n_trials=cli.trials,
        catch=[torch.cuda.OutOfMemoryError],
        gc_after_trial=True,
    )
