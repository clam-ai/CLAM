# pylint: disable=C0413
"""
Script for finding the optimal hyperparameters of an optimization technique
using the `Optuna` library
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
from sscompiler.compiler import AbstractTransformer
from sscompiler.compiler.layers import (
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
    mark_adapters_as_trainable,
)
from sscompiler.utils.argument_classes import (
    ExperimentOptions,
    SlimscaleParser,
    ValidationOptions,
)
from sscompiler.utils.constants import BASE_DIR, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, get_hyperparams, nf4
from sscompiler.utils.tokenization import DATALOADER_MAP
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

TEST_DIR = os.path.join(BASE_DIR, "compiler/tests")


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
    validation = args.validation
    experiment = args.experiment

    trials = get_hyperparams(validation.database)

    if "mistralai" in experiment.model:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tokenizer = MistralTokenizer.v3()
    else:
        tokenizer = AutoTokenizer.from_pretrained(experiment.model)
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train, tokenized_test = DATALOADER_MAP[experiment.task](tokenizer, "test")

    target_modules = TARGET_MODULES[experiment.model]
    auto_config = AutoConfig.from_pretrained(
        experiment.model,
        num_labels=2,
        finetuning_task=experiment.task,
    )

    auto_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=experiment.model,
        torch_dtype=torch.bfloat16,
        config=auto_config,
        ignore_mismatched_sizes=True,
        device_map="auto",
    )
    auto_model.config.pad_token_id = auto_model.config.eos_token_id
    at = AbstractTransformer(
        model_dir=experiment.model,
        groups=target_modules,
        auto_model=auto_model,
    )

    if experiment.quantize:
        nf4(at)

    mark_adapters_as_trainable(at.auto_model)
    total_memory_mb, total_params, trainable_params = at.print_trainable_parameters()

    i, trial = trials.popitem()

    logger.info("--------------------------------")
    logger.info("Trial %d", i)
    at = AbstractTransformer(
        model_dir=experiment.model,
        groups=target_modules,
        auto_model=auto_model,
        device_map="auto",
    )

    if experiment.quantize:
        nf4(at)

    learning_rate = trial.pop("learning_rate", 1e-4)
    logger.info("Trial %d Learning Rate: %f", i, learning_rate)
    opts = trial["opts"]
    scale_factor = trial.get("scale_factor", 1)
    logger.info("Scale factor is: %f", scale_factor)

    for j, o in enumerate(opts):
        opt = o.pop("opt")
        logger.info(
            "Opt %d: Applying %s to model with hyperparameters: %s", j + 1, opt, o
        )
        OPT_MAP[opt](at, scale_factor=scale_factor, **o)

    logger.info("Trial: %d", trial.number)
    logger.info("Parameters: %s", trial.params)
    logger.info("Total Memory Size: %.2f MB", total_memory_mb)
    logger.info("Total Parameters: %d", total_params)
    logger.info("Trainable Parameters: %d", trainable_params)

    try:
        final_score, history = finetune_at(
            at=at,
            task="cve",
            tokenized_train=tokenized_train,
            tokenized_eval=tokenized_test,
            epochs=experiment.epochs,
            batch_size=experiment.batch_size,
            train_head=experiment.train_head,
            metric_names=["f1", "accuracy"],
            use_multi_lr=True,
            tokenizer=tokenizer,
        )
        logger.info(history)
    except torch.cuda.OutOfMemoryError as oom:
        logger.error(oom)
        torch.cuda.empty_cache()
        raise oom

    final_score = max(
        history,
        key=lambda i, name="eval_combined_score": (
            i["eval_combined_score"] if "eval_combined_score" in i else -1
        ),
    )["eval_combined_score"]

    return final_score, trainable_params


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_arguments(ValidationOptions, dest="validation")
    parser.add_arguments(ExperimentOptions, dest="experiment")
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
    for seed in [0, 10, 100, 1000, 10000]:
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
