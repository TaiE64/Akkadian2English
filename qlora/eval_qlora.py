#!/usr/bin/env python3
"""
Evaluate QLoRA adapter vs baseline on validation set.
Compares: baseline (no adapter) vs adapter-merged model.

Usage:
    conda activate kaggle_deep_past
    python qlora/eval_qlora.py
    python qlora/eval_qlora.py --adapter_dir qlora/qlora_adapter_final
"""

import argparse
import os
import sys
import math
import time
import warnings
from pathlib import Path
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# ============================================================
# chrF++ (inline, from notebook)
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


def compute_scores(preds, refs):
    """Compute corpus-level chrF++ and optionally BLEU via sacrebleu."""
    chrf_scores = [_chrfpp_sentence_score(p, r) for p, r in zip(preds, refs)]
    chrf = np.mean(chrf_scores) if chrf_scores else 0.0

    # Try sacrebleu for BLEU + official chrF++
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        chrf_official = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
        geo = math.sqrt(max(bleu, 0) * max(chrf_official, 0))
        return {
            "bleu": round(bleu, 2),
            "chrf_official": round(chrf_official, 2),
            "chrf_inline": round(chrf, 2),
            "geomean": round(geo, 2),
        }
    except ImportError:
        return {"chrf_inline": round(chrf, 2)}


def generate_predictions(model, tokenizer, texts, device, max_new_tokens=256, num_beams=1):
    """Generate predictions for a list of texts."""
    results = []
    model.eval()
    with torch.inference_mode():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text, max_length=512, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                use_cache=True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(decoded)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{len(texts)}...")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/byt5-xl-akkadian")
    parser.add_argument("--adapter_dir", type=str, default="qlora/qlora_adapter_final")
    parser.add_argument("--data_dir", type=str, default="qlora/prepared_data")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use all val samples")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline eval")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load validation data
    val_path = Path(args.data_dir) / "val.csv"
    if not val_path.exists():
        print(f"ERROR: {val_path} not found. Run prepare_data.py first!")
        sys.exit(1)

    val_df = pd.read_csv(val_path, encoding="utf-8")
    if args.max_samples > 0:
        val_df = val_df.head(args.max_samples)

    texts = val_df["input_text"].tolist()
    refs = val_df["tgt_processed"].tolist()
    print(f"Validation samples: {len(texts)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ═══ Baseline evaluation ═══
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print("BASELINE (no adapter, fp16)")
        print(f"{'='*60}")
        print("Loading base model in fp16...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device).eval()

        t0 = time.time()
        baseline_preds = generate_predictions(
            base_model, tokenizer, texts, device, num_beams=args.num_beams
        )
        baseline_time = time.time() - t0
        baseline_scores = compute_scores(baseline_preds, refs)
        print(f"  Time: {baseline_time:.1f}s ({baseline_time/len(texts):.2f}s/sample)")
        print(f"  Scores: {baseline_scores}")

        # Free memory
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        baseline_scores = None
        baseline_preds = None

    # ═══ Adapter evaluation ═══
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        print(f"\nAdapter not found at {adapter_dir}. Skipping adapter eval.")
        return

    print(f"\n{'='*60}")
    print("ADAPTER (merged, fp16)")
    print(f"{'='*60}")

    # Load base model + merge adapter (fp16, not quantized)
    print("Loading base model + adapter...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("Merging adapter...")
    model = model.merge_and_unload()
    model = model.to(device).eval()

    t0 = time.time()
    adapter_preds = generate_predictions(
        model, tokenizer, texts, device, num_beams=args.num_beams
    )
    adapter_time = time.time() - t0
    adapter_scores = compute_scores(adapter_preds, refs)
    print(f"  Time: {adapter_time:.1f}s ({adapter_time/len(texts):.2f}s/sample)")
    print(f"  Scores: {adapter_scores}")

    # ═══ Comparison ═══
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    if baseline_scores and "geomean" in baseline_scores and "geomean" in adapter_scores:
        diff = adapter_scores["geomean"] - baseline_scores["geomean"]
        print(f"  Baseline GeoMean: {baseline_scores['geomean']}")
        print(f"  Adapter  GeoMean: {adapter_scores['geomean']}")
        print(f"  Difference:       {diff:+.2f}")
        if diff > 0:
            print(f"\n  >>> IMPROVEMENT! Proceed with merge and Kaggle submission.")
        else:
            print(f"\n  >>> No improvement. Consider tuning hyperparameters.")
    elif baseline_scores:
        print(f"  Baseline: {baseline_scores}")
        print(f"  Adapter:  {adapter_scores}")

    # Show example differences
    print(f"\nExample outputs (first 5):")
    for i in range(min(5, len(texts))):
        print(f"\n  Input: {texts[i][:80]}...")
        print(f"  Ref:   {refs[i][:80]}")
        if baseline_preds:
            print(f"  Base:  {baseline_preds[i][:80]}")
        print(f"  Adapt: {adapter_preds[i][:80]}")


if __name__ == "__main__":
    main()
