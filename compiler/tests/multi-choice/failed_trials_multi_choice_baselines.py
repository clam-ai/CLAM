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
from sscompiler.utils.tokenization import DATALOADER_MAP
from transformers import AutoConfig, AutoModelForSequenceClassification

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests/multi-choice")
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
    experiment = args.experiment
    baseline = args.baselines.baseline

    if "t5" in experiment.model or (
        "gemma" in args.model and experiment.task == "stsb"
    ):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    target_modules = TARGET_MODULES[experiment.model]
    auto_config = AutoConfig.from_pretrained(
        experiment.model,
        num_labels=4,
        finetuning_task=experiment.task,
    )

    def objective(trial):
        auto_model = AutoModelForSequenceClassification.from_pretrained(
            experiment.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            config=auto_config,
            ignore_mismatched_sizes=True,
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

        if "gemma" in args.model:
            tokenizer = at.get_tokenizer(
                add_bos_token=True,
                add_eos_token=True,
                pad_token="eos",
                padding_side="right",
            )
        else:
            tokenizer = at.get_tokenizer()

        tokenized_train, tokenized_eval = DATALOADER_MAP[experiment.task](
            tokenizer=tokenizer,
            validation_set="validation",
            padding=experiment.should_pad,
            max_length=experiment.max_length,
        )

        try:
            _, history = finetune_at(
                at=at,
                task=experiment.task,
                tokenizer=tokenizer,
                tokenized_train=tokenized_train,
                tokenized_eval=tokenized_eval,
                epochs=experiment.epochs,
                batch_size=experiment.batch_size,
                train_head=experiment.train_head,
                metric_names=["accuracy", "f1"],
                use_multi_lr=False,
                learning_rate=learning_rate,
            )
            logger.info(history)
        except ValueError as ve:
            logger.error(ve)
            logger.error(trial.params)
            torch.cuda.empty_cache()
            return -1

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

    assert experiment.task in ["hellaswag", "mmlu", "arc-e", "arc-c"]

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
        f"failed_epochs[{experiment.epochs}]"
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

    study = optuna.load_study(
        study_name=study_name,
        storage=SQLITE_DB,
    )

    logger.info("Rerun failed trials.")
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    for failed_trial in failed_trials:
        logger.info("Rerunning trial %d", failed_trial.number)
        try:
            study.enqueue_trial(failed_trial.params)
        except Exception as e:
            logger.error("Failed to enqueue trial %d: %s", failed_trial.number, str(e))

    study.optimize(
        create_objective(cli),
        n_trials=len(failed_trials),
        gc_after_trial=True,
        catch=[torch.cuda.OutOfMemoryError],
    )
