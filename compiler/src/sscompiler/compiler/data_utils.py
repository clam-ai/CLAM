import functools
import os
import random

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset


def prepare_calibration_input(
    model, dataloader, device, seqlen=1024, is_decoder_only=True
):
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    elif is_decoder_only:
        layers = model.model.layers
    else:
        encoder_layers = model.model.encoder.layers
        decoder_layers = model.model.decoder.layers
    # # dev = model.hf_device_map["model.embed_tokens"]
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]

    seqlen = 1024

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (128, seqlen, model.config.hidden_size),
        dtype=dtype,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            if "OPT" not in model.__class__.__name__:
                cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    if is_decoder_only:
        layers[0] = Catcher(layers[0])
    else:
        encoder_layers[0] = Catcher(encoder_layers[0])
        decoder_layers[0] = Catcher(decoder_layers[0])

    for batch in dataloader:
        try:
            if is_decoder_only:
                model(batch[0].to(device))
        except ValueError:
            pass

    if is_decoder_only:
        layers[0] = layers[0].module
        layers[0] = layers[0].to(device)
    else:
        encoder_layers[0] = encoder_layers[0].module
        decoder_layers[0] = decoder_layers[0].module
        encoder_layers[0] = encoder_layers[0].to(device)
        decoder_layers[0] = decoder_layers[0].to(device)

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    return inps, outs, attention_mask, position_ids


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_loader_from_dataset(dataset, nsamples, seqlen, tokenizer):
    enc = tokenizer(" ".join(dataset), return_tensors="pt")
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        "allenai/c4", split="train", data_files="en/c4-train.00001-of-01024.json.gz"
    )
    data_files = {"validation": "en/c4-validation.*.json.gz"}

    valdata = load_dataset("allenai/c4", data_files=data_files, split="validation")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def preprocess_squad(tokenizer):
    train_dataset, eval_dataset = load_dataset(
        "rajpurkar/squad_v2",
        split=[
            "train[:10]",
            "validation[:10]",
        ],
    )
    column_names = train_dataset.column_names

    preprocess_train_with_tokenizer = functools.partial(_preprocess_train, tokenizer)
    processed_train = train_dataset.map(
        preprocess_train_with_tokenizer,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )

    preprocess_eval_with_tokenizer = functools.partial(_preprocess_eval, tokenizer)
    processed_eval = eval_dataset.map(
        preprocess_eval_with_tokenizer,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

    return processed_train, processed_eval


def _preprocess_train(tokenizer, examples):
    """
    Processes the training data for Question-Answer datasets so the model can be trained on it
    """
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["context"],
        examples["question"],
        truncation="only_first",
        max_length=1024,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            token_end_index = 0
            while sequence_ids[token_start_index] != 0:
                token_end_index += 1

            token_end_index = len(input) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def _preprocess_eval(tokenizer, examples):
    """
    Processes the evaluation data for Question-Answer datasets so the model can be evaluated
    """
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["context"],
        examples["question"],
        truncation="only_first",
        max_length=1024,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples
