import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8, 9"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(".")
import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from sscompiler.compiler.data_utils import get_loaders
from sscompiler.compiler import (
    AbstractTransformer,
)

from sscompiler.utils.constants import (
    SUPERGLUE_DATASETS,
    METRIC_MAP,
    TARGET_MODULES,
    TASK_TO_KEYS,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from sscompiler.compiler.eval_utils import opt_eval, llama_eval


def main(args):
    model = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, device_map="auto")
    # model=model.to_bettertransformer()

    # model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token

    auto_model = AutoModelForCausalLM.from_pretrained(
        args.path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for dataset in ["wikitext2", "ptb", "c4"]:
        # for dataset in ['c4']:
        # for dataset in ['wikitext2']:
        dataloader, testloader = get_loaders(dataset, seed=0, tokenizer=tokenizer)
        print(dataset)
        if "opt" in args.base_model:
            opt_eval(auto_model, testloader, "cuda", dataset)
        elif (
            "huggyllama" in args.base_model
            or "Llama" in args.base_model
            or "llama" in args.base_model
        ):
            llama_eval(auto_model, testloader, "cuda", dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Eval Script")
    parser.add_argument(
        "--base_model",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Pretrained model ID",
    )
    args = parser.parse_args()

    main(args)
    """
    --base_model=meta-llama/Llama-2-70b-hf --path=outputs/meta-llama/Llama-2-70b-hf/0.09999999999999998%S_PTQ
    """
