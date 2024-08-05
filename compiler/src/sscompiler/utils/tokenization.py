"""
This package includes functions for tokenizing all datasets available in our repo
"""

import functools
import os
from typing import Dict, Tuple, Union

import numpy as np

# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from datasets import Dataset, DatasetDict, load_dataset

from .constants import (
    BASE_DIR,
    GLUE_DATASETS,
    SUPERGLUE_DATASETS,
    TASK_TO_KEYS,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def preprocess_boolq(row):
    return f"{row["passage"]}\nQuestion: {row["question"]}?\nAnswer:"


def preprocess_wic(row):
    return f"wic sentence1:{row['sentence1']} sentence2:{row['sentence2']} word:{row['word']}"


def preprocess_cb(row):
    return f"cb hypothesis:{row['hypothesis']} premise:{row['premise']}"


def preprocess_copa(row):
    question = row["question"]
    premise = row["premise"]
    choice1 = row["choice1"]
    choice2 = row["choice2"]
    arg = f"copa choice1:{choice1} choice2:{choice2} premise:{premise} question:{question}"
    return arg


# prompt templates taken from this paper / repo
# https://arxiv.org/pdf/1910.10683 (google-research/text-to-text-transfer-transformer)
SUPERGLUE_PROCESSORS = {
    "wic": preprocess_wic,
    "cb": preprocess_cb,
    "copa": preprocess_copa,
    "boolq": preprocess_boolq,
}


def tokenize_glue(
    tokenizer: PreTrainedTokenizer,
    task: str,
    model: PreTrainedModel,
    should_pad: bool,
    max_length: int,
    full_train=False,
):
    """
    Tokenizes a GLUE (or SuperGLUE) dataset for a given task
    """
    if full_train:
        split = [
            "train",
            "validation_matched" if task == "mnli" else "validation",
        ]
    else:
        split = [
            f"train[:{TRAIN_SPLIT[task]}%]",
            f"train[-{VAL_SPLIT[task]}%:]",
        ]

    if task == "boolq":
        train_dataset, eval_dataset = load_dataset(
            "google/boolq",
            split=split,
        )
    else:
        train_dataset, eval_dataset = load_dataset(
            "super_glue" if task in SUPERGLUE_DATASETS else "glue",
            task,
            split=split,
        )

    assert isinstance(train_dataset, Dataset)
    assert isinstance(eval_dataset, Dataset)

    # this is from the LoftQ study for preprocessing the dataset, see here:
    # https://github.com/yxli2123/LoftQ/blob/ed5ba19e285b598109c7915586434d81e3d34748/glue/run_glue.py#L323
    is_regression = task == "stsb"
    label_key = "answer" if task == "boolq" else "label"
    if task == "boolq":
        num_labels = 2
        label_list = [0, 1]
    elif not is_regression:
        label_list = train_dataset.features[label_key].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
    elif not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # TODO: analyze the coverage of these cases better
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in model.config.label2id.items()
        }
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in model.config.label2id.items()
        }

    sentence1, sentence2 = TASK_TO_KEYS.get(task, (None, None))

    # this is also from LoftQ, but is almost the exact same as the setup from
    # Amazon's design spaces paper
    def preprocess_function(examples):
        # Tokenize the texts
        if task in SUPERGLUE_PROCESSORS:
            args_ex = SUPERGLUE_PROCESSORS[task](examples)
            result = tokenizer(
                args_ex["inputs"] if task == "wsc" else args_ex,
                padding=should_pad,
                max_length=max_length,
                truncation=True,
            )
        else:
            args_ex = (
                (examples[sentence1],)
                if sentence2 is None
                else (examples[sentence1], examples[sentence2])
            )
            result = tokenizer(
                *args_ex, padding=should_pad, max_length=max_length, truncation=True
            )

        if label_key in examples:
            result["labels"] = examples[label_key]
            if task == "boolq":
                result["labels"] = np.array(result["labels"], dtype=int).tolist()
        else:
            raise RuntimeError("Labels not found.")
        return result

    processed_train = train_dataset.map(
        preprocess_function,
        batched=task in GLUE_DATASETS,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train",
    )
    processed_eval = eval_dataset.map(
        preprocess_function,
        batched=task in GLUE_DATASETS,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on eval",
    )

    return processed_train, processed_eval


def tokenize_arc(
    *,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    version: str = "ARC-Easy",
    validation_set: str = "validation",
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    assert version in [
        "ARC-Easy",
        "ARC-Challenge",
    ], f"Invalid ARC version type {version}"
    assert validation_set in [
        "validation",
        "test",
    ], f"Invalid validation set {validation_set}"

    raw_dataset = load_dataset(
        "allenai/ai2_arc",
        name=version,
        trust_remote_code=True,
    )

    keys_to_labels = {"1": 0, "A": 0, "2": 1, "B": 1, "3": 2, "C": 2, "4": 3, "D": 3}

    def preprocess_function(examples):
        questions = examples["question"]
        options = examples["choices"]
        answers = examples["answerKey"]

        inputs = []
        labels = []

        for question, options, answer in zip(questions, options, answers):
            input_text = f"Question: {question}\n\nChoices:\n"

            # skip examples with more than 5 options
            if "E" in options["label"]:
                continue
            for key, value in zip(options["label"], options["text"]):
                key = keys_to_labels[key]
                input_text += f"{key}: {value}\n"
            input_text += "\nAnswer: "

            inputs.append(input_text)
            labels.append(keys_to_labels[answer])

        return {"input_text": inputs, "label": labels}

    processed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    if "should_pad" in kwargs:
        should_pad = kwargs.pop("should_pad")

    def tokenize_function(examples):
        result = tokenizer(
            examples["input_text"],
            truncation=True,
            **kwargs,
        )

        result["labels"] = examples["label"]

        return result

    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_dataset["train"].column_names,
    )

    return tokenized_dataset["train"], tokenized_dataset[validation_set]


def tokenize_hellaswag(
    *,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validation_set: str = "validation",
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    assert validation_set in [
        "validation",
        "test",
    ], f"Invalid validation set {validation_set}"

    raw_train, raw_eval = load_dataset(
        "Rowan/hellaswag",
        split=[
            "train",
            "validation",
        ],
    )

    def preprocess_function(examples):
        if len(examples["label"]) == 0:
            return {}

        input_text = f"{examples['activity_label']}\n\n" f"{examples['ctx']}\n\n"

        for i, choice in enumerate(examples["endings"]):
            input_text += f"{i}: {choice}\n"
        input_text += "\nAnswer: "

        return {"input_text": input_text, "label": int(examples["label"])}

    processed_train = raw_train.map(
        preprocess_function,
        remove_columns=raw_train.column_names,
    )
    processed_eval = raw_eval.map(
        preprocess_function,
        remove_columns=raw_train.column_names,
    )

    if "should_pad" in kwargs:
        should_pad = kwargs.pop("should_pad")

    def tokenize_function(examples):
        result = tokenizer(
            examples["input_text"],
            truncation=True,
            **kwargs,
        )
        result["labels"] = examples["label"]
        return result

    tokenized_train = processed_train.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_train.column_names,
    )
    tokenized_eval = processed_eval.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_train.column_names,
    )

    if validation_set == "validation":
        # pass in seed to make sure we get the same split every time
        split = tokenized_train.train_test_split(test_size=0.6, seed=0)
        tokenized_train, tokenized_eval = split["train"], split["test"]

    return tokenized_train, tokenized_eval


MMLU_FEWSHOT = """
Question: What is the capital of France?
0: Berlin
1: Madrid
2: Paris
3: Rome
Answer: 2

Question: What is the largest planet in our solar system?
0: Earth
1: Mars
2: Jupiter
3: Saturn
Answer: 2

Question: What is the powerhouse of the cell?
0: Nucleus
1: Mitochondria
2: Ribosome
3: Endoplasmic Reticulum
Answer: 1
"""


def tokenize_mmlu(
    *,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validation_set: str = "validation",
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    assert validation_set in [
        "validation",
        "test",
    ], f"Invalid validation set {validation_set}"

    raw_dataset = load_dataset("cais/mmlu", "all")
    assert isinstance(raw_dataset, DatasetDict)

    def preprocess_function(examples):
        questions = examples["question"]
        choices = examples["choices"]
        answers = examples["answer"]

        inputs = []
        labels = []

        for question, choice, answer in zip(questions, choices, answers):
            prompt = f"Question: {question}\n\nChoices:\n"
            for label, option in enumerate(choice):
                prompt += f"{label}: {option}\n"
            prompt += "\nAnswer: "
            prompt = MMLU_FEWSHOT + prompt
            inputs.append(prompt)
            labels.append(answer)

        return {"input_text": inputs, "label": labels}

    processed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset["auxiliary_train"].column_names,
    )

    def tokenize_function(examples):
        result = tokenizer(
            examples["input_text"],
            truncation=True,
            **kwargs,
        )
        result["labels"] = examples["label"]
        return result

    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_dataset["auxiliary_train"].column_names,
    )

    tokenized_dataset["train"] = tokenized_dataset["auxiliary_train"]

    if validation_set == "validation":
        tokenized_dataset = (
            tokenized_dataset["train"]
            .shuffle(seed=0)
            .train_test_split(
                train_size=0.05,
                test_size=0.10,
                seed=0,
            )
        )
        validation_set = "test"

    return tokenized_dataset["train"], tokenized_dataset[validation_set]


def tokenize_primevul(
    *,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validation_set: str = "validation",
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    assert validation_set in [
        "validation",
        "test",
    ], f'Invalid key: "{validation_set}".'

    data_dir = os.path.join(BASE_DIR, "datasets", "PrimeVul_Data/cleaned")
    raw_dataset = load_dataset(
        "json",
        data_dir=data_dir,
    )
    assert isinstance(raw_dataset, DatasetDict)

    max_tokens = 2048

    def tokenize_example(examples):
        result = {}

        # Codestral is weird so need to preprocess data differently
        if isinstance(tokenizer, MistralTokenizer):
            input_ids = []
            for example in examples["code"]:
                completion_request = ChatCompletionRequest(
                    messages=[UserMessage(content=example)]
                )
                tokens = tokenizer.encode_chat_completion(completion_request).tokens
                input_ids.append(tokens)

            result["input_ids"] = input_ids
        else:
            result = tokenizer(
                examples["code"],
                truncation=True,
                **kwargs,
            )

        result["labels"] = examples["label"]

        return result

    tokenized_dataset = raw_dataset.map(
        tokenize_example,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) < max_tokens,
        batched=False,
        desc="Filtering dataset",
    )

    return tokenized_dataset["train"], tokenized_dataset[validation_set]


def tokenize_cve(
    *,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validation_set: str = "validation",
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    raw_data = load_dataset(
        "json",
        data_dir=os.path.join(BASE_DIR, "datasets/cpp_vulnerabilities"),
    )
    assert isinstance(raw_data, DatasetDict)
    train_data = raw_data["train"]
    test_data = raw_data["test"]

    def preprocess(examples):
        result = {}
        input_ids = []

        if isinstance(tokenizer, MistralTokenizer):
            for example in examples["code"]:
                completion_request = ChatCompletionRequest(
                    messages=[UserMessage(content=example)]
                )
                tokens = tokenizer.encode_chat_completion(completion_request).tokens
                input_ids.append(tokens)

            result["input_ids"] = input_ids
        else:
            for example in examples["code"]:
                result = tokenizer(
                    example,
                    padding=False,
                    truncation=True,
                    **kwargs,
                )
                input_ids.append(result["input_ids"])

        result["labels"] = 1 if examples["label"] else 0
        return result

    tokenized_train = train_data.map(
        preprocess,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on train",
    )
    tokenized_test = test_data.map(
        preprocess,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on eval",
    )

    if validation_set == "validation":
        # pass in seed to make sure we get the same split every time
        split = tokenized_train.train_test_split(test_size=0.2, seed=0)
        tokenized_train, tokenized_test = split["train"], split["test"]

    return tokenized_train, tokenized_test


DATALOADER_MAP = {
    "arc-e": functools.partial(tokenize_arc, version="ARC-Easy"),
    "arc-c": functools.partial(tokenize_arc, version="ARC-Challenge"),
    "mmlu": tokenize_mmlu,
    "hellaswag": tokenize_hellaswag,
    "primevul": tokenize_primevul,
    "cve": tokenize_cve,
}
