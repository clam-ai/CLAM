"""
Script for finding the optimal hyperparameters of an optimization technique
using the `Optuna` library
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import gc
import json
import logging
import sqlite3

import jsbeautifier
import lm_eval
import optuna
import setproctitle
import torch
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
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    T5ForSequenceClassification,
)

from datasets import load_dataset

options = jsbeautifier.default_options()
options.indent_size = 2

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
STORAGE_DIR = "/srv/shared_home/common-data/slimscale/slimscale-databases"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TASK_DIR = os.path.join(BASE_DIR, "tasks")


global_hyperparams = {}


def ia3(trial, at, opt_num):
    extra_args = {}
    ia3_lr = global_hyperparams["ia3"]["learning_rate"]
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


def lora(trial, at, opt_num):
    lora_r = global_hyperparams["lora"]["r"]
    lora_lr = global_hyperparams["lora"]["learning_rate"]
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


def loha(trial, at, opt_num):
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


def vera(trial, at, opt_num):
    vera_r = global_hyperparams["vera"]["r"]
    vera_lr = global_hyperparams["vera"]["learning_rate"]
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


def loftq(trial, at: AbstractTransformer, opt_num, config):
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
    all_others = [i for i in config["target_modules"].keys() if i not in vq]
    at.inject_adapter(all_others, lambda x: Portable4BitLinear.from_linear(x))


def fp4(trial, at: AbstractTransformer, opt_num, config):
    at.inject_adapter(list(config.keys()), lambda x: Portable4BitLinear.from_linear(x))


OPT_MAP = {
    "ia3": ia3,
    "lora": lora,
    "loha": loha,
    "loha": loha,
    "vera": vera,
}
QUANTIZE_MAP = {
    # "loftq": loftq,
    "fp4": fp4,
}


def create_objective(
    config, task, epochs=5, num_opts=1, model_name="google-t5/t5-base"
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
    raw_datasets = load_dataset(
        "super_glue" if task in SUPERGLUE_DATASETS else "glue",
        task,
    )
    is_regression = task == "stsb"
    if not is_regression:
        num_labels = len(raw_datasets["train"].features["label"].names)
    else:
        num_labels = 1
    del raw_datasets
    gc.collect()
    torch.cuda.empty_cache()

    def objective(trial):
        gc.collect()
        torch.cuda.empty_cache()
        auto_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task=task,
        )
        if "t5" in model_name:
            auto_model = T5ForSequenceClassification.from_pretrained(
                model_name,
                device_map="cuda",
                config=auto_config,
                ignore_mismatched_sizes=True,
            )
        elif "gemma" in model_name and task == "stsb":
            auto_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map="cuda",
                config=auto_config,
                ignore_mismatched_sizes=True,
            )
        else:
            auto_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                config=auto_config,
                ignore_mismatched_sizes=True,
            )
        at = AbstractTransformer(
            model_name,
            TARGET_MODULES[model_name],
            auto_model=auto_model,
        )

        if args.prune:
            wanda(at, raw_datasets["train"], task, **args.__dict__)
        quant = ""

        if args.quantize:
            quant = trial.suggest_categorical("quant", QUANTIZE_MAP.keys())
            # special case where loftq is also considered the first peft opt. since it introduces lora
            QUANTIZE_MAP[quant](trial, at, 0, config)
        gc.collect()
        torch.cuda.empty_cache()
        opts = []
        for i in range(1 if "loftq" == quant else 0, num_opts):
            opt_i = trial.suggest_categorical(f"opt_{i}", OPT_MAP.keys())
            opts.append(opt_i)
        for i, opt in enumerate(opts):
            OPT_MAP[opt](trial, at, i)

        mark_adapters_as_trainable(at.auto_model)
        if args.train_head:
            if "gemma" in model_name:
                for name, param in at.auto_model.named_parameters():
                    if "score" in name:
                        param.requires_grad = True

        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info(f"Trial: {trial.number}")
        logger.info(f"Parameters: {trial.params}")
        logger.info(f"Total Memory Size: {total_memory_mb:.2f} MB")
        logger.info(f"Total Parameters: {total_params}")
        logger.info(f"Trainable Parameters: {trainable_params}")

        final_score = finetune_at(at, task, use_multi_lr=True, **args.__dict__)

        return final_score

    return objective


def process_query(database):

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Slimscale Optuna Search",
        description="Performs hyperparameter search over Slimscale optimizations",
    )
    parser.add_argument(
        "--task",
        default="cb",
        choices=METRIC_MAP.keys(),
        help="LLM task to optimize over",
    )
    parser.add_argument(
        "--model_name", default="google-t5/t5-base", help="Hugging Face model"
    )
    parser.add_argument(
        "--database_dir",
        default="/home/ubuntu/slimscale/compiler/tests/databases/t5_cb_baselines_dbs",
        help="Directory of optuna dbs to query over",
    )
    # can be 5 for larger datasets
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_opts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    # can be 5 for larger datasets
    parser.add_argument("--max_length", type=int, default=512)
    # should be no for T5, yes for gemma
    parser.add_argument("--should_pad", type=bool, default=False)
    # should be no for T5, yes for gemma
    parser.add_argument("--should_pad", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--prune", type=str, default=False)
    parser.add_argument("--sparsity_ratio", type=int, default=0.5)
    parser.add_argument("--structured", type=bool, default=False)
    # should be no for T5, yes for experiment 2
    parser.add_argument("--train-head", type=bool, default=True)
    parser.add_argument("--debug-mode", type=bool, default=False)

    args = parser.parse_args()

    setproctitle.setproctitle(
        f"Slimscale ({args.task}) -O{args.num_opts}"
        + (" DEBUG" if args.debug_mode else "")
    )

    databases = [f for f in os.listdir(args.database_dir) if f.endswith(".db")]
    for database in databases:
        opt = (database.split("opt_[")[-1]).split("]")[0]
        database = os.path.join(args.database_dir, database)
        global_hyperparams[opt] = process_query(database)

    model_config = TARGET_MODULES[args.model_name]

    if args.should_pad and "t5" in args.model_name:
        raise RuntimeError("Should not pad T5.")

    model_name = (args.model_name).split("/")[-1]
    study_name = f"part_a_model_[{model_name}]_task_[{args.task}]_trial_[{args.trials}]_epochs[{args.epochs}]_num_opts_[{args.num_opts}]_batch_size[{args.batch_size}]_max_length_[{args.max_length}]_train_head_[{args.train_head}]_should_pad_[{args.should_pad}]_quantized_[{args.quantize}]"
    if args.debug_mode:
        study_name = "debug"

    log_file = f"{study_name}.log"

    logger.addHandler(logging.FileHandler(log_file))
    logger.info("\n--------------------------------")
    logger.info(f"Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"Using seed: {seed}")
    logger.info(f"Using global hyperparameters: {global_hyperparams}")
    logger.info(f"Using global hyperparameters: {global_hyperparams}")

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    direction = "maximize"
    if args.task == "wikitext":
        direction = "minimize"

    additional_args = {}

    db_dir = os.path.join(
        STORAGE_DIR,
        model_name,
        "experiment-1" if args.quantize is False else "experiment-2",
        "part-a",
        f"{args.num_opts}-opts",
    )
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{study_name}.db")
    SQLITE_DB = f"sqlite:///{db_path}"

    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
        direction=direction,
        study_name=study_name,
        storage=SQLITE_DB,
        load_if_exists=True,
        **additional_args,
    )

    logger.info("Start optimization.")
    task = args.__dict__.pop("task")
    study.optimize(
        create_objective(
            model_config,
            task=task,
            epochs=args.epochs,
            num_opts=args.num_opts,
            model_name=args.model_name,
        ),
        n_trials=args.trials,
    )
