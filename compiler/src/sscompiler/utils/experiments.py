import json
import os
import sqlite3
from itertools import chain
from typing import List, Union

import evaluate
import numpy as np
from accelerate import Accelerator
from bitsandbytes.optim import AdamW
from sscompiler.compiler import AbstractTransformer, Portable4BitLinear, WandaLayer
from sscompiler.compiler.layers.peft import collect_all_peft_params
from sscompiler.compiler.multilr import MultiLR
from sscompiler.utils.constants import METRIC_MAP, SUPERGLUE_DATASETS, TASK_TO_KEYS
from sscompiler.utils.tokenization import SUPERGLUE_PROCESSORS
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from ..compiler.layers.peft import mark_adapters_as_trainable


def get_hyperparams(database):
    """
    Gets the hyperparameters for an abstract transformer from a sqlite database.
    """
    database = os.path.join(database)
    con = sqlite3.connect(database)
    cur = con.cursor()

    query = """
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
    if len(rows) < 1:
        raise ValueError(
            "No rows were returned by the query. Check that the database contains entries for this task"
        )

    # creates dictionary of (trial, optimizations) pairs
    # optimizations contains the learning rate used in the trial, and an ordered
    # list of optimizations with their hyperparameters
    trials = {}
    for row in rows:
        trial_id, param, value, dist = row
        dist = json.loads(dist)
        trial = trials.get(trial_id, {})
        trial_opts = trial.get("opts", [])
        if param == "quant":
            continue
        if dist["name"] == "CategoricalDistribution":
            value = dist["attributes"]["choices"][int(value)]
        if param == "learning_rate":
            trial["learning_rate"] = float(value)
            trials[trial_id] = trial
            continue
        if param == "scale_factor":
            trial["scale_factor"] = float(value)
            trials[trial_id] = trial
            continue
        params = param.rsplit("_", 1)
        param, opt_num = params[0], int(params[1])
        if len(trial_opts) <= opt_num:
            trial_opts.extend([{} for _ in range(opt_num + 1 - len(trial_opts))])
        opt_dict = trial_opts[opt_num]
        opt_dict[param] = value
        trial["opts"] = trial_opts
        trials[trial_id] = trial

    return trials


def finetune_at(
    *,
    at: AbstractTransformer,
    task: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokenized_train,
    tokenized_eval,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    train_head: bool = False,
    metric_names: Union[List[str], None] = None,
    use_multi_lr: bool = False,
    mode: str = "Train",
):
    """
    Fine-tunes an abstract transformer on a specified task.
        at:     AbstractTransformer model
        task:   name of the task; be sure to include the task name and
                appropriate dataset tokenizing method to DATASET_MAP
    """
    mark_adapters_as_trainable(at.auto_model)
    if train_head:
        for name, param in at.auto_model.named_parameters():
            if "score" in name:
                param.requires_grad = True

    at.auto_model = Accelerator().prepare(at.auto_model)

    output_dir = os.path.join("testing")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.06,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        optim="paged_adamw_32bit",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    if metric_names is None:
        metric_names = METRIC_MAP[task]
        if not isinstance(metric_names, list):
            metric_names = [metric_names]
    metrics = [evaluate.load(metric) for metric in metric_names]

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task == "stsb" else np.argmax(preds, axis=1)
        result = dict(
            chain.from_iterable(
                metric.compute(
                    predictions=preds,
                    references=p.label_ids,
                    **({"average": "macro"} if metric.name == "f1" else {}),
                ).items()
                for metric in metrics
            )
        )
        result["combined_score"] = np.mean(list(result.values())).item()
        return result

    optimizer_parameters = collect_all_peft_params(at.auto_model)

    optimizer = AdamW(optimizer_parameters)

    # round up to the nearest multiple of 8 b/c hugggingface pads
    total_steps = int(epochs * (((len(tokenized_train) + 7) // 8) * 8) / batch_size)

    lamdbda_factories = []
    for _ in optimizer_parameters:
        lamdbda_factories.append(
            lambda y: get_linear_schedule_with_warmup(y, 0, total_steps)
        )

    multilr_scheduler = MultiLR(optimizer, lambda_factories=lamdbda_factories)

    if use_multi_lr:
        optimizers = (optimizer, multilr_scheduler)
    else:
        optimizers = (None, None)

    trainer = Trainer(
        model=at.auto_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
    )
    trainer.train()

    metric_names = [f"eval_{name}" for name in metric_names]

    best = [
        max(
            trainer.state.log_history,
            key=lambda i, name=name: i[name] if name in i else -1,
        ).get(name, -1)
        for name in metric_names
    ]

    return best, trainer.state.log_history


def wanda(at: AbstractTransformer, train_dataset, task):
    """
    Prunes an abstract transformer using Wanda.
    """
    at.inject_adapter(
        [key for key, _ in at.groups.items()],
        lambda x: WandaLayer(
            x,
            sparsity_ratio=0.5,
        ),
    )

    sentences = []
    if task in SUPERGLUE_DATASETS:
        sentences = [
            SUPERGLUE_PROCESSORS[task](row) for row in train_dataset
        ]  # switch to map
    else:
        sentence1, sentence2 = TASK_TO_KEYS.get(task, (None, None))
        for example in train_dataset:
            args_ex = (
                (example[sentence1],)
                if sentence2 is None
                else (example[sentence1], example[sentence2])
            )
            sentences.extend(args_ex)
    at.prune(sentences)

    # at.check_sparsity()


def glue_activations(at: AbstractTransformer, train_dataset, task):
    """
    Prunes an abstract transformer using Wanda.
    """

    sentences = []
    if task in SUPERGLUE_DATASETS:
        sentences = [
            SUPERGLUE_PROCESSORS[task](row) for row in train_dataset
        ]  # switch to map
    else:
        sentence1, sentence2 = TASK_TO_KEYS.get(task, (None, None))
        for example in train_dataset:
            args_ex = (
                (example[sentence1],)
                if sentence2 is None
                else (example[sentence1], example[sentence2])
            )
            sentences.extend(args_ex)
    return at.get_activations(sentences)


def nf4(at: AbstractTransformer):
    """
    Quantizes an abstract transformer using 4-bit quantization.
    """
    print("Quantizing model")
    at.inject_adapter(
        list(at.groups.keys()), lambda layer: Portable4BitLinear.from_linear(layer)
    )
