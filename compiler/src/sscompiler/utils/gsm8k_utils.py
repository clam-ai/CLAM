# pylint: disable=C0413
import os
import re
from copy import deepcopy
from typing import Union

import torch
from accelerate import Accelerator
from bitsandbytes.optim import AdamW
from sscompiler.compiler import AbstractTransformer, mark_adapters_as_trainable
from sscompiler.compiler.layers.peft import collect_all_peft_params
from sscompiler.compiler.multilr import MultiLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from datasets import Dataset, DatasetDict, load_dataset

device = "cuda"


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance.

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


def extract_output(pred, trigger=""):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ""
    output = pred[start + len(trigger) :].lstrip()  # left strip any whitespaces
    return output


def compute_metrics(
    intervenable,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    data_items: list,
    trigger_tokens: str,
    batch_size: int = 4,
):
    # switch the tokenizer mode first for generation tasks

    # tokenizer.padding_side = "left"  # switch padding side for collator

    # eval_dataset["ids"] = torch.arange(0, len(eval_dataset))
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=intervenable,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size, collate_fn=data_collator, shuffle=False
    )
    correct_count = 0
    total_count = 0
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)

    if (
        "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path
    ):  # pretty bad workaround for llama-3, forgive me
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        trigger_tokens = "assistant\n\n"

    id = 0
    with torch.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # get left padding count, [batch_size], and add to locations
            left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(
                as_tuple=True
            )[1]
            if left_padding.numel() > 0:
                left_padding = left_padding.reshape(1, -1, 1).to(
                    device
                )  # [1, batch_size, 1]
            else:
                print("Warning: No BOS token found, skipping left padding adjustment.")

            # set generation args depending on task
            generation_args = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }

            generation_args.update(
                {
                    "max_new_tokens": 256,
                    "do_sample": False,
                }
            )

            if (
                "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path
            ):  # pretty bad workaround for llama-3, forgive me
                generation_args["eos_token_id"] = terminators

            # generate with intervention on prompt
            steered_response = intervenable.generate(**generation_args)

            # detokenize in batch
            actual_preds = tokenizer.batch_decode(
                steered_response, skip_special_tokens=True
            )

            for pred in actual_preds:
                example = data_items[total_count]
                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except:
                    print("get not split based on trigger tokens: ", raw_generation)
                    raw_generation = "WRONG"

                # check if generation is correct
                answer = example["answer"].split("####")[-1].strip()
                generation = extract_answer_number(raw_generation)
                if abs(float(extract_answer_number(answer)) - generation) <= 0.001:
                    correct_count += 1

                # log
                total_count += 1

                metric_str = round(correct_count / total_count, 3)
                eval_iterator.set_postfix({"em": metric_str})

    return correct_count / total_count


class GSM8KCallBack(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset,
        data_items,
        trigger_tokens,
        batch_size,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_items = data_items
        self.trigger_tokens = trigger_tokens
        self.batch_size = batch_size

    def on_epoch_end(self, args, state, control, **kwargs):
        result = compute_metrics(
            self.model,
            self.tokenizer,
            self.eval_dataset,
            self.data_items,
            self.trigger_tokens,
            self.batch_size,
        )
        state.log_history.append({"eval_result": result})


def tokenize_gsm8k(
    *,
    tokenizer: PreTrainedTokenizer,
    validation_set: str = "validation",
    **kwargs,
):
    assert validation_set in [
        "validation",
        "test",
    ], "please enter a valid `validation_set`"

    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

    raw_dataset = load_dataset(
        "openai/gsm8k",
        "main",
    )
    assert isinstance(raw_dataset, DatasetDict), "Error loading dataset"

    def tokenize_function(data_item):
        result = {}
        if (
            "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path
        ):  # pretty bad workaround for llama-3, forgive me
            system_prompt = "You are a helpful assistant."
            # we remove the BOS, otherwise there will be redundant BOS tokens.
            base_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data_item["question"]},
                ],
                tokenize=False,
            )[len("<|begin_of_text|>") :]
            base_input = (
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data_item["question"]},
                        {"role": "assistant", "content": data_item["answer"]},
                    ],
                    tokenize=False,
                )[len("<|begin_of_text|>") :]
                + tokenizer.eos_token
            )
        else:  # setup is from https://github.com/yxli2123/LoftQ/
            base_prompt = f"{data_item['question']}{QUESTION_PROMPT}"
            # note: we remove the extra space here to keep the format clean.
            base_input = (
                base_prompt
                + f"{data_item['answer']}{tokenizer.eos_token}".replace(
                    "####", "The final answer is: "
                )
            )

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=kwargs.get("max_length", 1024),
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input,
            max_length=kwargs.get("max_length", 1024),
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]

        output_ids = deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = -100

        result["input_ids"] = base_input_ids
        result["labels"] = output_ids
        return result

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=raw_dataset["train"].column_names,
    )

    if validation_set == "validation":
        percent_train = 0.10
        percent_eval = 0.01
        train_size = int(percent_train * len(tokenized_dataset["train"]))
        eval_size = int(percent_eval * len(tokenized_dataset["train"]))
        train_set = tokenized_dataset["train"].select(range(0, train_size))
        eval_set = tokenized_dataset["train"].select(
            range(
                len(tokenized_dataset["train"]) - eval_size,
                len(tokenized_dataset["train"]),
            )
        )
        untokenized_train_set = raw_dataset["train"].select(range(0, train_size))
        untokenized_eval_set = raw_dataset["train"].select(
            range(
                len(tokenized_dataset["train"]) - eval_size,
                len(tokenized_dataset["train"]),
            )
        )
        return (train_set, eval_set), (untokenized_train_set, untokenized_eval_set)
    else:
        return (
            (tokenized_dataset["train"], tokenized_dataset[validation_set]),
            (raw_dataset["train"], raw_dataset[validation_set]),
        )


def finetune_gsm8k(
    *,
    at: AbstractTransformer,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokenized_train,
    tokenized_eval,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    train_head: bool = False,
    use_multi_lr: bool = False,
    untokenized_eval: Dataset,
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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=at, label_pad_token_id=-100, padding="longest"
    )

    mark_adapters_as_trainable(at.auto_model)

    at.print_trainable_parameters()

    at.auto_model.train()

    # # training args
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

    trigger_tokens = "First think step by step and then answer the final number.\n"
    callback = GSM8KCallBack(
        at.auto_model,
        tokenizer,
        tokenized_eval,
        untokenized_eval,
        trigger_tokens,
        batch_size,
    )

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

    # make trainer
    trainer = Trainer(
        model=at.auto_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[callback],
        optimizers=optimizers,
    )

    trainer.train()

    history = trainer.state.log_history

    final_score = max(
        history,
        key=lambda i, name="eval_result": (
            i["eval_result"] if "eval_result" in i else -1
        ),
    )["eval_result"]

    return final_score, history
