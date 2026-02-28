#!/usr/bin/env python3
"""
Merge QLoRA adapter into base model for Kaggle deployment.
Output: full fp16 model that can replace the original on Kaggle.

Usage:
    conda activate kaggle_deep_past
    python qlora/merge_adapter.py
    python qlora/merge_adapter.py --adapter_dir qlora/qlora_adapter_final --output_dir qlora/merged_model
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge QLoRA adapter into base model")
    parser.add_argument("--model_path", type=str, default="models/byt5-xl-akkadian")
    parser.add_argument("--adapter_dir", type=str, default="qlora/qlora_adapter_final")
    parser.add_argument("--output_dir", type=str, default="qlora/merged_model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        print(f"ERROR: Adapter not found at {adapter_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model in bfloat16 on CPU (matches training compute dtype)
    print(f"Loading base model from {args.model_path} (bfloat16, CPU)...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Match training compute dtype
        device_map="cpu",
    )
    print(f"  Base params: {sum(p.numel() for p in base_model.parameters()):,}")

    # Load adapter
    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # Merge and unload
    print("Merging adapter into base model...")
    model = model.merge_and_unload()
    print(f"  Merged params: {sum(p.numel() for p in model.parameters()):,}")

    # Convert to float16 for Kaggle (standard inference format)
    print("Converting to float16 for Kaggle deployment...")
    model = model.half()

    # Save
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(output_dir)

    # Report size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\nMerged model saved: {total_size / 1e9:.2f} GB")
    print(f"Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

    print(f"\nDone! Upload {output_dir} to Kaggle as a dataset.")
    print("Then update the notebook model_path to point to the new dataset.")


if __name__ == "__main__":
    main()
