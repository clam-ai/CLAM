import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(".")
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sscompiler.compiler.evaluater import evaluate_model


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.path, device_map="auto", torch_dtype=torch.float16
    )
    # Quick evaluate
    evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "boolq,piqa,hellaswag,winogrande",
        limit=200,
        eval_ppl="wikitext2,ptb,c4",
    )
    """
    # MMLU evaluate
    evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu",
        limit=-1,
        eval_ppl="",
    )
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--path",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-2b",
        help="Pretrained model ID",
    )
    args = parser.parse_args()

    main(args)
