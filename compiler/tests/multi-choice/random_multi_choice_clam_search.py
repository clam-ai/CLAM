"""
Script for finding the optimal hyperparameters of an optimization technique
using the `Optuna` library
"""

# pylint: disable=C0413
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
    Portable4BitLinear,
    PortableIA3Adapter,
    PortableLoftQLayer,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
    mark_adapters_as_trainable,
)
from sscompiler.utils.constants import METRIC_MAP, SUPERGLUE_DATASETS, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, wanda
from sscompiler.utils.tokenization import tokenize_glue
from torch.cuda import OutOfMemoryError
from transformers import AutoConfig, AutoModelForSequenceClassification

options = jsbeautifier.default_options()
options.indent_size = 2

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests")
STORAGE_DIR = "/srv/shared_home/common-data/slimscale/slimscale-databases"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


global_hyperparams = {
    "loha": {"learning_rate": 0.004},
    "vera": {"learning_rate": 0.1},
    "ia3": {"learning_rate": 0.04},
    "lora": {"learning_rate": 0.0004},
}


def ia3(trial, at, opt_num, scaling_factor):
    ia3_lr = global_hyperparams["ia3"]["learning_rate"] * scaling_factor
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


def loftq(trial, at: AbstractTransformer, opt_num):
    loftq_r = trial.suggest_categorical(f"loftq_r_{opt_num}", [4, 8, 16, 32, 64])
    vq = ["value", "query"]
    at.inject_adapter(
        vq,
        lambda x: PortableLoftQLayer(
            x,
            r=loftq_r,
            num_bits=4,
            num_iter=5,
        ),
    )
    all_others = [i for i in at.groups.keys() if i not in vq]
    at.inject_adapter(all_others, lambda x: Portable4BitLinear(x))


def fp4(trial, at: AbstractTransformer, opt_num):
    at.inject_adapter(at.groups.keys(), Portable4BitLinear)


OPT_MAP = {
    "ia3": ia3,
    "lora": lora,
    "loha": loha,
    "vera": vera,
}
QUANTIZE_MAP = {
    # "loftq": loftq,
    "fp4": fp4,
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
    model_name = args.model

    auto_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=4,
        finetuning_task=task,
    )

    if "t5" in args.model or ("gemma" in args.model and args.task == "stsb"):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    def objective(trial):
        gc.collect()
        torch.cuda.empty_cache()
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

        quant = ""
        if args.quantize:
            quant = trial.suggest_categorical("quant", QUANTIZE_MAP.keys())
            # special case where loftq is also considered the first peft opt. since it introduces lora
            QUANTIZE_MAP[quant](trial, at, 0)

        gc.collect()
        torch.cuda.empty_cache()

        scale_factor = trial.suggest_categorical("scale_factor", [0.1, 0.4, 1, 4])

        opts = []
        for i in range(1 if "loftq" == quant else 0, args.num_opts):
            opt_i = trial.suggest_categorical(f"opt_{i}", OPT_MAP.keys())
            opts.append(opt_i)
        for i, opt in enumerate(opts):
            OPT_MAP[opt](trial, at, i, scale_factor)
        mark_adapters_as_trainable(at.auto_model)
        if args.train_head:
            if "gemma" in model_name:
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

        try:
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
        except OutOfMemoryError as oom:
            logger.error(oom)
            logger.error(trial.params)
            torch.cuda.empty_cache()
            gc.collect()
            raise oom
        except Exception as e:
            logger.error(e)
            raise e

        return result

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Slimscale Optuna Search",
        description="Performs hyperparameter search over Slimscale optimizations",
    )
    parser.add_argument(
        "--task",
        choices=METRIC_MAP.keys(),
        help="LLM task to optimize over",
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=TARGET_MODULES.keys(),
        help="Hugging Face model",
        required=True,
    )
    # can be 5 for larger datasets
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_opts", type=int, default=3)
    parser.add_argument("--trials", type=int, default=17000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    # should be no for T5, yes for gemma
    parser.add_argument("--should_pad", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--prune", type=str, default=False)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--structured", type=bool, default=False)
    # should be no for T5, yes for experiment 2
    parser.add_argument("--train_head", type=bool, default=False)
    parser.add_argument("--debug-mode", type=bool, default=False)

    args = parser.parse_args()

    setproctitle.setproctitle(
        f"Random Slimscale ({args.task}) Part B Order {args.num_opts}"
    )

    if args.should_pad and "t5" in args.model:
        raise RuntimeError("Should not pad T5.")

    model_name = (args.model).split("/")[-1]

    # name of the optuna study. should include all hyperparameters for easy querying / regex operations
    study_name = (
        f"random_model_[{model_name}]"
        f"_task_[{args.task}]"
        f"_num_opts_[{args.num_opts}]"
        f"_quantized_[{args.quantize}]"
        f"_trial_[{args.trials}]"
        f"_epochs[{args.epochs}]"
        f"_batch_size[{args.batch_size}]"
        f"_max_length_[{args.max_length}]"
        f"_train_head_[{args.train_head}]"
        f"_should_pad_[{args.should_pad}]"
    )
    if args.debug_mode:
        study_name = "debug"

    log_dir = os.path.join(
        TEST_DIR,
        "logs",
        "random",
        "part-b",
        model_name,
        "experiment-1" if args.quantize is False else "experiment-2",
        args.task,
        f"{args.num_opts}-opts",
    )

    log_name = (
        f"epochs[{args.epochs}]"
        f"_batch_size[{args.batch_size}]"
        f"_max_length_[{args.max_length}]"
        f"_train_head_[{args.train_head}]"
        f"_should_pad_[{args.should_pad}]"
        f"_quantized_[{args.quantize}]"
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

    direction = "minimize" if args.task == "wikitext" else "maximize"

    db_dir = os.path.join(
        STORAGE_DIR,
        "random",
        model_name,
        "experiment-1" if args.quantize is False else "experiment-2",
        "part-b",
        f"{args.num_opts}-opts",
    )
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{study_name}.db")
    SQLITE_DB = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=SQLITE_DB,
        load_if_exists=True,
        sampler=optuna.samplers.BruteForceSampler(),
    )

    logger.info("Start optimization.")
    study.optimize(
        create_objective(
            args,
        ),
        n_trials=args.trials,
        gc_after_trial=True,
        catch=[OutOfMemoryError],
    )
