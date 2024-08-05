# pylint: disable=C0413
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(".")

import argparse
import logging

import torch
from accelerate import Accelerator
from sscompiler.compiler import AbstractTransformer
from sscompiler.compiler.data_utils import get_loaders, get_qat_dataset
from sscompiler.compiler.eval_utils import llama_eval, opt_eval
from sscompiler.compiler.layers.pbllm2 import GPTQLayer
from sscompiler.compiler.layers.peft import (
    mark_adapters_as_trainable,
    pbllm_mark_adapters_as_trainable,
)
from sscompiler.utils.constants import TARGET_MODULES
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset, load_dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def to_regular_linear(root_module):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, GPTQLayer):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            linear = module.dequantize()
            setattr(father, name[ind + 1 :], linear)
            print(f"replace layer {name} with {linear}")


def gptq(at: AbstractTransformer, train_dataset, mode, low_frac):
    at.inject_adapter(
        # ["dense1", "dense2"],
        # ["dense1"],
        list(at.groups.keys()),
        adapter_fn=lambda x: GPTQLayer.from_linear(
            x,
            8,
            low_frac,
            mode=mode,
        ),
    )
    if mode == "PTQ":
        at.quant(train_dataset)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("prepare training data")
    if args.mode == "QAT":
        raw_datasets = get_qat_dataset(args.dataset, tokenizer, args.data_percent)
    else:
        raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    assert isinstance(raw_datasets, Dataset)

    auto_config = AutoConfig.from_pretrained(args.model, num_labels=1)

    auto_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=auto_config,
        ignore_mismatched_sizes=True,
    )

    at = AbstractTransformer(
        model_dir=args.model,
        groups=TARGET_MODULES[args.model],
        auto_model=auto_model,
    )

    total_memory_mb, total_params, trainable_params = at.print_trainable_parameters()
    if args.mode == "PTQ":
        mark_adapters_as_trainable(at.auto_model)
        at.auto_model.eval()

    gptq(at, raw_datasets["text"], args.mode, args.low_frac)

    if args.mode == "QAT":
        pbllm_mark_adapters_as_trainable(at.auto_model)
        at.auto_model = Accelerator().prepare(at.auto_model)

        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )

        logger.info("Total Memory Size: %.2f MB", total_memory_mb)
        logger.info("Total Parameters: %d", total_params)
        logger.info("Trainable Parameters: %d", trainable_params)

        # train
        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=args.train_steps * 0.05,
            max_steps=args.train_steps,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_torch",
            report_to="tensorboard",
            save_safetensors=False,
        )

        # Create trainer
        trainer = Trainer(
            model=at.auto_model,
            train_dataset=raw_datasets,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        at.auto_model.config.use_cache = False

        # Train the model
        trainer.train()
        print(f"Trainer state log history: {str(trainer.state.log_history)}")

        at.auto_model.eval()
        salient_frac = 1 - args.low_frac
        save_dir = f"outputs/{args.base_model}/{salient_frac}%S_QAT_{args.train_steps}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        to_regular_linear(at.auto_model)
        at.auto_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"model saved to {save_dir}")
    else:
        total_memory_mb, total_params, trainable_params = (
            at.print_trainable_parameters()
        )
        # PTQ mode        #
        # save partially binarized model for future use

        save_dir = f"outputs/{args.model}/{args.mode}_{args.low_frac}NS"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # this may not be necessary
        at.prep_for_save()
        at.auto_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"model saved to {save_dir}")
        # tokenizer = AutoTokenizer.from_pretrained(args.model) this line was already stated above
        for dataset in ["wikitext2", "ptb", "c4"]:
            # for dataset in ['c4']:
            # for dataset in ['wikitext2']:
            packed = get_loaders(dataset, seed=seed, tokenizer=tokenizer)
            assert packed is not None
            _, testloader = packed
            if "opt" in args.model:
                opt_eval(at.auto_model, testloader, "cuda", dataset)
            elif (
                "huggyllama" in args.model
                or "Llama" in args.model
                or "llama" in args.model
            ):
                llama_eval(at.auto_model, testloader, "cuda", dataset)

        # at.auto_model.eval()


# where all the argument parsing should happen
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to optimize and search over",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="QAT",
        choices=["PTQ", "QAT"],
    )

    parser.add_argument("--low_frac", type=float, default=0.7)

    parser.add_argument("--baseline", type=str, default="gptq", help="Baseline")
    parser.add_argument(
        "-s", "--train_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--dataset", type=str, default="red_pajama", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=10, help="Percentage of data to use"
    )

    args = parser.parse_args()

    main(args)
