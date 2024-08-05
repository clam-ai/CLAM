"""
Searches a SQLite database of optuna trials for the top performing
configurations on a task, and then trains a model with these configurations to
completion
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import json
import logging
import sqlite3
import time

import evaluate
import numpy as np
import setproctitle
import torch
from accelerate import Accelerator
from datasets import load_dataset
from sscompiler.compiler import AbstractTransformer
from sscompiler.compiler.layers import (
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
    mark_adapters_as_trainable,
)
from sscompiler.compiler.multilr import MultiLR
from sscompiler.utils.constants import METRIC_MAP, SUPERGLUE_DATASETS, TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, get_hyperparams, nf4, wanda
from sscompiler.utils.tokenization import tokenize_glue
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    T5ForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

global_hyperparams = {}


logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TASK_DIR = os.path.join(BASE_DIR, "tasks")
LOG_FILE = os.path.join(TASK_DIR, ".tmp/full_train.log")
logging.basicConfig(filename=LOG_FILE, encoding="utf8", level=logging.INFO)


def ia3(at, **kwargs):
    ia3_lr = global_hyperparams["ia3"]["learning_rate"]
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


def lora(at):
    lora_r = global_hyperparams["lora"]["r"]
    lora_lr = global_hyperparams["lora"]["learning_rate"]
    at.inject_adapter(
        ["value", "query"],
        lambda x: PortableLoRAAdapter(
            x,
            r=lora_r,
            # lora_alpha=lora_alpha,
            bias=False,
            in_features=x.in_features,
            out_features=x.out_features,
            lora_lr=lora_lr,
        ),
    )


def loha(at):
    loha_r = global_hyperparams["loha"]["r"]
    loha_lr = global_hyperparams["loha"]["learning_rate"]
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


def vera(at, **kwargs):
    vera_r = global_hyperparams["vera"]["r"]
    vera_lr = global_hyperparams["vera"]["learning_rate"]
    # if not args.brute:
    #     extra_args["vera_lr"] = trial.suggest_float(f"vera_lr_{opt_num}", 4e-3, 1e-2, log=True)
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


def bitfit(at):
    at.inject_adapter(
        ["query", "gate"],
        # BitFit
    )


OPT_MAP = {"ia3": ia3, "lora": lora, "loha": loha, "vera": vera, "bitfit": bitfit}


def get_global_hyperparams(database):
    con = sqlite3.connect(database)
    cur = con.cursor()

    query = f"""
        WITH
        top_trials AS (
            SELECT t.trial_id, t.study_id, tv.value
            FROM (select * from trials WHERE state='COMPLETE' order by trial_id asc limit 100) t
            JOIN trial_values tv ON t.trial_id=tv.trial_id
            ORDER BY tv.value DESC
            LIMIT 1
        )
        SELECT t.trial_id, tp.param_name, tp.param_value, tp.distribution_json
        FROM top_trials t
        JOIN trial_params tp ON t.trial_id=tp.trial_id;
    """

    cur.execute(query)
    rows = cur.fetchall()

    trial = {}
    for row in rows:
        trial_id, param, value, dist = row
        dist = json.loads(dist)

        if dist["name"] == "CategoricalDistribution":
            value = dist["attributes"]["choices"][int(value)]
        if param.startswith("r_"):
            trial["r"] = int(value)
        if param == "learning_rate":
            trial["learning_rate"] = float(value)
    return trial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_argument(
        "--task", type=str, help="The name of the GLUE task", default="cb"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="/home/ubuntu/slimscale/compiler/tests/part_a_model_[t5-base]_task_[cb]_trial_[16]_epochs[10]_num_opts_[2]_batch_size[8]_max_length_[512]_train_head_[False]_should_pad_[False]_quantized_[False].db",
        help="Absolute path to the SQLite database to query",
    )
    parser.add_argument(
        "--database_dir",
        type=str,
        default="/home/ubuntu/slimscale/compiler/tests/databases/t5_cb_baselines_dbs",
        help="Absolute path to the SQLite database directory containing the databases of the results of the baseline",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="The name of the base model to train",
        default="google-t5/t5-base",
    )
    parser.add_argument("--seed", type=int, help="The torch / random seed", default=0)
    parser.add_argument("--batch_size", type=int, default=8)
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
    setproctitle.setproctitle(f"Slimscale, ({args.task}), order 2, Top 1 Model")

    databases = [f for f in os.listdir(args.database_dir) if f.endswith(".db")]
    for database in databases:
        opt = (database.split("opt_[")[-1]).split("]")[0]
        database = os.path.join(args.database_dir, database)
        global_hyperparams[opt] = get_global_hyperparams(database)

    trials = get_hyperparams(args.database)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Using seed: {seed}")
    logger.info(f"Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
        logger.info("Trial %d Learning Rate: %f", i, learning_rate)
        opts = trial["opts"]

        start = time.time()
        for j, o in enumerate(opts):
            # opt = o.pop('opt')
            logger.info(
                "Opt %d: Applying %s to model with optimal hyperparameters",
                j + 1,
                o["opt"],
            )
            OPT_MAP[o["opt"]](at)
        end = time.time()
        logger.info(f"Optimization time: {end - start}")

        task = args.__dict__.pop("task")
        metrics = finetune_at(
            at,
            task,
            learning_rate=learning_rate,
            full_train=True,
            use_multi_lr=True,
            **args.__dict__,
        )
        logger.info(metrics)
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
