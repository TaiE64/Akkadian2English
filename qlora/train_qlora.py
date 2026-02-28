#!/usr/bin/env python3
"""
QLoRA fine-tuning script for ByT5-XL Akkadian translation.

Usage:
    conda activate kaggle_deep_past
    python qlora/prepare_data.py          # Run first to prepare data
    python qlora/train_qlora.py           # Then run this
    python qlora/train_qlora.py --lr 1e-4 --epochs 3 --rank 16  # Custom params
"""

import argparse
import os
import sys
import math
import warnings
from pathlib import Path
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# Inline chrF++ (from notebook, no sacrebleu dependency)
# ============================================================
def _extract_char_ngrams(s, n):
    return Counter([s[i:i+n] for i in range(len(s) - n + 1)])

def _extract_word_ngrams(s, n):
    words = s.split()
    return Counter([tuple(words[i:i+n]) for i in range(len(words) - n + 1)])

def _chrfpp_sentence_score(hypothesis, reference, char_order=6, word_order=2, beta=2.0):
    if not hypothesis or not reference:
        return 0.0
    total_f = 0.0
    count = 0
    for n in range(1, char_order + 1):
        h_ngrams = _extract_char_ngrams(hypothesis, n)
        r_ngrams = _extract_char_ngrams(reference, n)
        if not h_ngrams or not r_ngrams:
            continue
        common = sum((h_ngrams & r_ngrams).values())
        prec = common / max(sum(h_ngrams.values()), 1)
        rec = common / max(sum(r_ngrams.values()), 1)
        if prec + rec > 0:
            f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        else:
            f = 0.0
        total_f += f
        count += 1
    for n in range(1, word_order + 1):
        h_ngrams = _extract_word_ngrams(hypothesis, n)
        r_ngrams = _extract_word_ngrams(reference, n)
        if not h_ngrams or not r_ngrams:
            continue
        common = sum((h_ngrams & r_ngrams).values())
        prec = common / max(sum(h_ngrams.values()), 1)
        rec = common / max(sum(r_ngrams.values()), 1)
        if prec + rec > 0:
            f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        else:
            f = 0.0
        total_f += f
        count += 1
    return (total_f / max(count, 1)) * 100.0


# ============================================================
# Dataset
# ============================================================
class AkkadianQLoRADataset(Dataset):
    def __init__(self, df, tokenizer, max_source_len=512, max_target_len=256):
        self.tokenizer = tokenizer
        self.source_texts = df["input_text"].tolist()
        self.target_texts = df["tgt_processed"].tolist()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]

        # Don't pad here — let DataCollatorForSeq2Seq handle dynamic padding per batch
        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_len,
            truncation=True,
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
        )

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


# ============================================================
# Compute metrics
# ============================================================
def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Clip token IDs to valid range (ByT5 quantized model can produce out-of-range IDs)
        vocab_size = tokenizer.vocab_size  # 384 for ByT5
        preds = np.clip(preds, 0, vocab_size - 1)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels (replace -100 with pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Compute chrF++ (corpus-level average of sentence scores)
        chrf_scores = []
        for pred, ref in zip(decoded_preds, decoded_labels):
            chrf_scores.append(_chrfpp_sentence_score(pred, ref))
        chrf = np.mean(chrf_scores) if chrf_scores else 0.0

        # Also compute a rough BLEU via chrF++ (for reference only)
        # The main metric is chrF++
        return {
            "chrf": round(float(chrf), 2),
        }

    return compute_metrics


# ============================================================
# Main training function
# ============================================================
def train(args):
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")

    # Load prepared data
    data_dir = Path(args.data_dir)
    if not (data_dir / "train.csv").exists():
        print("ERROR: Prepared data not found. Run prepare_data.py first!")
        print(f"  Expected: {data_dir / 'train.csv'}")
        sys.exit(1)

    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(data_dir / "val.csv", encoding="utf-8")
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Quantization config
    print("\nSetting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model from {args.model_path} (4-bit NF4)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=args.dropout,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        bias="none",
    )

    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AkkadianQLoRADataset(
        train_df, tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )
    val_dataset = AkkadianQLoRADataset(
        val_df, tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Training arguments
    output_dir = args.output_dir
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,

        # Epochs & batching
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Learning rate
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,

        # Optimizer
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Precision
        bf16=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf",
        greater_is_better=True,

        # Generation for eval
        predict_with_generate=True,
        generation_max_length=384,
        generation_num_beams=1,  # greedy for speed

        # Logging
        logging_steps=args.logging_steps,
        report_to="none",

        # Memory
        gradient_checkpointing=True,
        dataloader_num_workers=0,

        # Misc
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting QLoRA training...")
    print(f"  LR: {args.lr}, Rank: {args.rank}, Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Eval every {args.eval_steps} steps")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final adapter
    adapter_dir = args.adapter_dir
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved to {adapter_dir}")

    # Final eval
    print("\nRunning final evaluation...")
    results = trainer.evaluate()
    print(f"Final eval results: {results}")

    print("\nDone! Next steps:")
    print(f"  1. Check results above")
    print(f"  2. Run: python qlora/eval_qlora.py")
    print(f"  3. If improved: python qlora/merge_adapter.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for ByT5-XL")
    parser.add_argument("--model_path", type=str, default="models/byt5-xl-akkadian")
    parser.add_argument("--data_dir", type=str, default="qlora/prepared_data")
    parser.add_argument("--output_dir", type=str, default="qlora/qlora_checkpoints")
    parser.add_argument("--adapter_dir", type=str, default="qlora/qlora_adapter_final")

    # Hyperparameters (easy to sweep)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=384)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()
    train(args)
