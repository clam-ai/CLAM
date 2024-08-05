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

import optuna
import setproctitle
import torch
from sscompiler.compiler import AbstractTransformer
from sscompiler.compiler.layers import (
    PortableIA3Adapter,
    PortableLoHAAdapter,
    PortableLoRAAdapter,
    PortableVeRAAdapter,
)
from sscompiler.utils.argument_classes import (
    BaselineOptions,
    ExperimentOptions,
    SlimscaleParser,
)
from sscompiler.utils.constants import TARGET_MODULES
from sscompiler.utils.experiments import finetune_at, nf4
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../"))

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


def load_data(tokenizer):
    raw_data: Dataset = load_dataset(
        "json",
        data_dir=os.path.join("data/cpp_vulnerabilities"),
    )["train"].train_test_split(test_size=0.2)
    train_data = raw_data["train"]
    test_data = raw_data["test"]

    def preprocess(examples):
        args_ex = (examples["code"],)
        result = tokenizer(*args_ex, padding=False, truncation=True)
        result["labels"] = 1 if examples["label"] else 0
        return result

    tokenized_train = train_data.map(
        preprocess,
        batched=False,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on train",
    )
    tokenized_test = test_data.map(
        preprocess,
        batched=False,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on eval",
    )

    return tokenized_train, tokenized_test


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
    experiment = args.experiment
    baseline = args.baselines.baseline

    tokenizer = AutoTokenizer.from_pretrained(experiment.model)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train, tokenized_test = load_data(tokenizer)

    target_modules = TARGET_MODULES[experiment.model]
    auto_config = AutoConfig.from_pretrained(
        experiment.model,
        num_labels=2,
        finetuning_task="cve",
    )

    def objective(trial):
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

        if args.quantize:
            nf4(at)

        gc.collect()
        torch.cuda.empty_cache()

        learning_rate = trial.suggest_categorical(
            "learning_rate", [1e-4, 4e-4, 1e-3, 4e-3, 1e-2, 4e-2, 1e-1]
        )
        opt_num = 1
        if baseline == "vera":
            r = trial.suggest_categorical(f"r_{opt_num}", [64, 128, 256, 512, 1024])
        elif baseline == "ia3":
            r = 0
        else:
            r = trial.suggest_categorical(f"r_{opt_num}", [4, 8, 16, 32, 64])
        OPT_MAP[baseline](trial, at, opt_num, r)

        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info("Trial: %d", trial.number)
        logger.info("Parameters: %s", trial.params)
        logger.info("Total Memory Size: %.2f MB", total_memory_mb)
        logger.info("Total Parameters: %d", total_params)
        logger.info("Trainable Parameters: %d", trainable_params)

        try:
            _, history = finetune_at(
                at=at,
                task=experiment.task,
                tokenizer=tokenizer,
                tokenized_train=tokenized_train,
                tokenized_eval=tokenized_test,
                epochs=experiment.epochs,
                batch_size=experiment.batch_size,
                train_head=experiment.train_head,
                metric_names=["accuracy", "f1"],
                use_multi_lr=False,
                learning_rate=learning_rate,
            )
            logger.info(history)
        except torch.cuda.OutOfMemoryError as oom:
            logger.error(oom)
            logger.error(trial.params)
            torch.cuda.empty_cache()
            raise oom

        final_score = max(
            history,
            key=lambda i, name="eval_combined_score": (
                i["eval_combined_score"] if "eval_combined_score" in i else -1
            ),
        )["eval_combined_score"]

        return final_score

    return objective


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_arguments(BaselineOptions, dest="baselines")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    cli = parser.parse_args()

    experiment = cli.experiment
    baselines = cli.baselines

    assert experiment.task == "cve"

    setproctitle.setproctitle(
        f"Slimscale {experiment.model}, {baselines.baseline}, {experiment.task}"
    )

    if experiment.should_pad and "t5" in experiment.model:
        raise RuntimeError("Should not pad T5.")

    model_name = (experiment.model).split("/")[-1]

    # name of the optuna study. should include all hyperparameters for easy querying / regex operations
    study_name = (
        f"model_[{model_name}]"
        f"_task_[{experiment.task}]"
        f"_opt_[{baselines.baseline}]"
        f"_quantized_[{experiment.quantize}]"
        f"_epochs[{experiment.epochs}]"
        f"_batch_size[{experiment.batch_size}]"
        f"_max_length_[{experiment.max_length}]"
        f"_train_head_[{experiment.train_head}]"
        f"_should_pad_[{experiment.should_pad}]"
    )

    log_dir = os.path.join(
        TEST_DIR,
        "logs",
        model_name,
        "experiment-1" if experiment.quantize is False else "experiment-2",
        "baselines",
        experiment.task,
        baselines.baseline,
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

    logger.info("\n--------------------------------")
    logger.info("Using devices: %s", os.environ["CUDA_VISIBLE_DEVICES"])
    logger.info("Using seed: %d", seed)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    db_dir = os.path.join(
        STORAGE_DIR,
        model_name,
        "experiment-1" if experiment.quantize is False else "experiment-2",
        "baselines",
        experiment.task,
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
        n_trials=40,
        gc_after_trial=True,
        catch=[torch.cuda.OutOfMemoryError],
    )
