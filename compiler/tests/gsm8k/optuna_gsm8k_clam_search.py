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
    mark_adapters_as_trainable,
)
from sscompiler.utils.argument_classes import (
    ExperimentOptions,
    SearchOptions,
    SlimscaleParser,
)
from sscompiler.utils.constants import TARGET_MODULES
from sscompiler.utils.experiments import nf4
from sscompiler.utils.gsm8k_utils import finetune_gsm8k, tokenize_gsm8k
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seed = 0
set_seed(seed)

TRAIN_DIR = os.path.join(BASE_DIR, "compiler/src/testing")
TEST_DIR = os.path.join(BASE_DIR, "compiler/tests/multi-choice")
STORAGE_DIR = "/srv/shared_home/common-data/slimscale/slimscale-databases"

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


def create_objective(args):
    experiment = args.experiment
    search = args.search

    model = args.model
    model_name = model
    num_opts = search.num_opts

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    (tokenized_train, tokenized_eval), (_, untokenized_eval) = tokenize_gsm8k(
        tokenizer=tokenizer,
        validation_set="validation",
        max_length=1024,
    )
    target_modules = TARGET_MODULES[experiment.model]

    def objective(trial):
        auto_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        at = AbstractTransformer(
            model_dir=experiment.model,
            groups=target_modules,
            auto_model=auto_model,
        )

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

        gc.collect()
        torch.cuda.empty_cache()

        mark_adapters_as_trainable(at.auto_model)
        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info("Trial: %d", trial.number)
        logger.info("Parameters: %s", trial.params)
        logger.info("Total Memory Size: %.2f MB", total_memory_mb)
        logger.info("Total Parameters: %d", total_params)
        logger.info("Trainable Parameters: %d", trainable_params)

        if need_resize:
            model.resize_token_embeddings(len(tokenizer))

        try:
            result, history = finetune_gsm8k(
                at=at,
                tokenizer=tokenizer,
                tokenized_train=tokenized_train,
                tokenized_eval=tokenized_eval,
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_head=args.train_head,
                use_multi_lr=True,
                untokenized_eval=untokenized_eval,
            )
            logger.info(history)
            logger.info(result)
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

        return result

    return objective


if __name__ == "__main__":
    parser = SlimscaleParser(
        description="Finetune best configurations to completion on a given task"
    )
    parser.add_arguments(SearchOptions, dest="search")
    parser.add_arguments(ExperimentOptions, dest="experiment")
    cli = parser.parse_args()

    experiment = cli.experiment
    search = cli.search

    setproctitle.setproctitle(
        f"Slimscale Part B {experiment.model}, {experiment.task}, Order {search.num_opts}"
    )

    if experiment.should_pad and "t5" in experiment.model:
        raise RuntimeError("Should not pad T5.")

    model_name = (experiment.model).split("/")[-1]

    # name of the optuna study. should include all hyperparameters for easy querying / regex operations
    study_name = (
        f"model_[{model_name}]"
        f"_task_[{experiment.task}]"
        f"_num_opts_[{search.num_opts}]"
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
        "part-b",
        model_name,
        "experiment-1" if experiment.quantize is False else "experiment-2",
        experiment.task,
        f"{search.num_opts}-opts",
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
    logger.info("Using hyperparameters: %s", global_hyperparams)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    direction = "maximize"

    db_dir = os.path.join(
        STORAGE_DIR,
        model_name,
        "experiment-1" if experiment.quantize is False else "experiment-2",
        "part-b",
        f"{search.num_opts}-opts",
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
        n_trials=search.trials,
        catch=[torch.cuda.OutOfMemoryError],
        gc_after_trial=True,
    )
