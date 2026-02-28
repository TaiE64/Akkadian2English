#!/usr/bin/env python3
"""
Test assisted generation vs beam search for ByT5-XL.
Compares quality (BLEU/chrF++) and speed.

Usage:
    conda activate kaggle_deep_past
    python test_assisted_gen.py --n_samples 30
"""

import argparse
import os
import time
import math
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import sacrebleu
except ImportError:
    os.system("pip install sacrebleu")
    import sacrebleu


def compute_scores(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    geo = math.sqrt(max(bleu.score, 0) * max(chrf.score, 0))
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2), "geomean": round(geo, 2)}


def run_generate(model, tokenizer, texts, device, **gen_kwargs):
    """Run generation on a list of texts, return decoded outputs and elapsed time."""
    results = []
    t0 = time.time()
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, max_length=384, truncation=True, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                use_cache=True,
                **gen_kwargs,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(decoded)
    elapsed = time.time() - t0
    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--model_path", type=str, default="models/byt5-xl-akkadian")
    parser.add_argument("--assistant_path", type=str, default="models/byt5-small")
    parser.add_argument("--data_path", type=str, default="data/competition/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print(f"Loading data from {args.data_path}")
    train_df = pd.read_csv(args.data_path, encoding="utf-8")
    valid = train_df.dropna(subset=["transliteration", "translation"])
    valid = valid[valid["translation"].str.strip() != ""]
    samples = valid.sample(n=min(args.n_samples, len(valid)), random_state=args.seed)

    texts = ["translate Akkadian to English: " + str(t) for t in samples["transliteration"].tolist()]
    refs = samples["translation"].tolist()
    print(f"Samples: {len(texts)}")

    # Load main model
    print(f"\nLoading main model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"Main model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load assistant model
    print(f"Loading assistant model: {args.assistant_path}")
    assistant = AutoModelForSeq2SeqLM.from_pretrained(args.assistant_path, torch_dtype=torch.float16)
    assistant = assistant.to(device).eval()
    print(f"Assistant model: {sum(p.numel() for p in assistant.parameters()):,} params")

    # Warm up
    print("\nWarming up...")
    _ = run_generate(model, tokenizer, texts[:1], device, num_beams=1)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══ Test 1: Beam search (num_beams=4) - current approach ═══
    print(f"\n{'='*60}")
    print("TEST 1: Beam search (num_beams=4) - CURRENT")
    print(f"{'='*60}")
    preds_beam4, time_beam4 = run_generate(
        model, tokenizer, texts, device,
        num_beams=4, length_penalty=1.3, early_stopping=True,
    )
    scores_beam4 = compute_scores(preds_beam4, refs)
    print(f"  Time: {time_beam4:.1f}s ({time_beam4/len(texts):.2f}s/sample)")
    print(f"  Scores: {scores_beam4}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══ Test 2: Greedy (num_beams=1) ═══
    print(f"\n{'='*60}")
    print("TEST 2: Greedy (num_beams=1)")
    print(f"{'='*60}")
    preds_greedy, time_greedy = run_generate(
        model, tokenizer, texts, device,
        num_beams=1,
    )
    scores_greedy = compute_scores(preds_greedy, refs)
    print(f"  Time: {time_greedy:.1f}s ({time_greedy/len(texts):.2f}s/sample)")
    print(f"  Scores: {scores_greedy}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══ Test 3: Greedy + Assisted (byt5-small) ═══
    print(f"\n{'='*60}")
    print("TEST 3: Greedy + Assisted (byt5-small)")
    print(f"{'='*60}")
    try:
        preds_assisted, time_assisted = run_generate(
            model, tokenizer, texts, device,
            num_beams=1,
            assistant_model=assistant,
        )
        scores_assisted = compute_scores(preds_assisted, refs)
        print(f"  Time: {time_assisted:.1f}s ({time_assisted/len(texts):.2f}s/sample)")
        print(f"  Scores: {scores_assisted}")

        # Verify outputs are identical to greedy
        match = sum(1 for a, b in zip(preds_greedy, preds_assisted) if a == b)
        print(f"  Match with greedy: {match}/{len(texts)} ({100*match/len(texts):.0f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
        time_assisted = None
        scores_assisted = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══ Test 4: Beam search (num_beams=2) - compromise ═══
    print(f"\n{'='*60}")
    print("TEST 4: Beam search (num_beams=2) - COMPROMISE")
    print(f"{'='*60}")
    preds_beam2, time_beam2 = run_generate(
        model, tokenizer, texts, device,
        num_beams=2, length_penalty=1.3, early_stopping=True,
    )
    scores_beam2 = compute_scores(preds_beam2, refs)
    print(f"  Time: {time_beam2:.1f}s ({time_beam2/len(texts):.2f}s/sample)")
    print(f"  Scores: {scores_beam2}")

    # ═══ Summary ═══
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'GeoMean':>8} {'Time':>8} {'Speedup':>8}")
    print("-" * 58)

    results = [
        ("Beam search (n=4) CURRENT", scores_beam4, time_beam4),
        ("Beam search (n=2)", scores_beam2, time_beam2),
        ("Greedy (n=1)", scores_greedy, time_greedy),
    ]
    if scores_assisted is not None:
        results.append(("Greedy + Assisted", scores_assisted, time_assisted))

    for name, scores, t in results:
        speedup = time_beam4 / t if t > 0 else 0
        print(f"{name:<30} {scores['geomean']:>8.2f} {t:>7.1f}s {speedup:>7.2f}x")

    # Quality vs Speed tradeoff
    print(f"\nQuality loss vs beam4:")
    for name, scores, t in results:
        diff = scores["geomean"] - scores_beam4["geomean"]
        speedup = time_beam4 / t if t > 0 else 0
        print(f"  {name:<30} GeoMean {diff:>+6.2f}  Speed {speedup:.2f}x")

    # Show some example differences
    print(f"\nExample outputs (first 3):")
    for i in range(min(3, len(texts))):
        print(f"\n  Input: {texts[i][:80]}...")
        print(f"  Ref:   {refs[i][:80]}...")
        print(f"  Beam4: {preds_beam4[i][:80]}...")
        print(f"  Grdy:  {preds_greedy[i][:80]}...")
        if scores_assisted is not None:
            print(f"  Asst:  {preds_assisted[i][:80]}...")


if __name__ == "__main__":
    main()
