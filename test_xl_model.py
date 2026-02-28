"""
Test ByT5-XL (4B) on competition train.csv with evaluation metrics.
Compare with ByT5-large (mattiaangeli mbr-v2) baseline.
"""
import torch
import os
import gc
import time
import re
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# Config
# ============================================================
XL_MODEL_PATH = r"C:\c\Users\29421\Desktop\kaggle_Challenge\models\byt5-xl-akkadian"
if not os.path.exists(XL_MODEL_PATH):
    XL_MODEL_PATH = r"C:\Users\29421\Desktop\kaggle_Challenge\models\byt5-xl-akkadian"

# mattiaangeli model (local if available)
LARGE_MODEL_PATH = None
for p in [
    r"C:\Users\29421\Desktop\kaggle_Challenge\models\byt5-akkadian-mbr-v2",
    r"C:\Users\29421\Desktop\kaggle_Challenge\models\mattiaangeli",
]:
    if os.path.exists(p):
        LARGE_MODEL_PATH = p
        break

TRAIN_CSV = r"C:\Users\29421\Desktop\kaggle_Challenge\data\competition\train.csv"
ORACC_CSV = r"C:\Users\29421\Desktop\kaggle_Challenge\data\manwithacat_oracc\train.csv"
NUM_EVAL_SAMPLES = 50  # evaluate on N samples
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
LENGTH_PENALTY = 1.3
BATCH_SIZE = 2
SEED = 42

# ============================================================
# Metrics: BLEU + chrF++ (inline, no sacrebleu needed)
# ============================================================
def _extract_char_ngrams(s, n):
    return Counter([s[i:i+n] for i in range(len(s) - n + 1)])

def _extract_word_ngrams(s, n):
    words = s.split()
    if len(words) < n:
        return Counter()
    return Counter([tuple(words[i:i+n]) for i in range(len(words) - n + 1)])

def chrfpp_score(hypothesis, reference, char_order=6, word_order=2, beta=2.0):
    """Sentence-level chrF++ score."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    total_f, count = 0.0, 0
    for n in range(1, char_order + 1):
        h_ng = _extract_char_ngrams(hypothesis, n)
        r_ng = _extract_char_ngrams(reference, n)
        if not h_ng or not r_ng:
            continue
        common = sum((h_ng & r_ng).values())
        p = common / max(sum(h_ng.values()), 1)
        r = common / max(sum(r_ng.values()), 1)
        f = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0.0
        total_f += f; count += 1
    for n in range(1, word_order + 1):
        h_ng = _extract_word_ngrams(hypothesis, n)
        r_ng = _extract_word_ngrams(reference, n)
        if not h_ng or not r_ng:
            continue
        common = sum((h_ng & r_ng).values())
        p = common / max(sum(h_ng.values()), 1)
        r = common / max(sum(r_ng.values()), 1)
        f = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0.0
        total_f += f; count += 1
    return (total_f / max(count, 1)) * 100.0

def bleu_score(hypothesis, reference, max_n=4):
    """Sentence-level BLEU (smoothed)."""
    hyp_words = hypothesis.strip().split()
    ref_words = reference.strip().split()
    if not hyp_words or not ref_words:
        return 0.0
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_words) / max(len(hyp_words), 1)))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = Counter([tuple(hyp_words[i:i+n]) for i in range(len(hyp_words) - n + 1)])
        ref_ng = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1)])
        common = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        # Add-1 smoothing
        prec = (common + 1) / (total + 1)
        log_avg += np.log(prec) / max_n
    return bp * np.exp(log_avg) * 100.0

def geomean(bleu, chrf):
    """Competition metric: geometric mean of BLEU and chrF++."""
    if bleu <= 0 or chrf <= 0:
        return 0.0
    return np.sqrt(bleu * chrf)

# ============================================================
# Preprocessing (same as 34.7 baseline)
# ============================================================
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})
_CHAR_MAP = {"ḫ":"h","Ḫ":"H","ʾ":"","₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
             "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","—":"-","–":"-"}
_CHAR_TRANS = str.maketrans(_CHAR_MAP)
_WS_RE = re.compile(r"\s+")
_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+|……|\[\.+\])")
_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))")
_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE = re.compile(r"(?<!\w)x(?!\w)", re.I)

def preprocess(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    s = s.replace("sz","š").replace("SZ","Š").replace("s,","ṣ").replace("S,","Ṣ").replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    s = _XTOKEN_RUN_RE.sub("<gap>", s)
    s = _ELLIPSIS_RE.sub("<gap>", s)
    s = _BRACKET_X_RE.sub("<gap>", s)
    s = _XRUN_RE.sub("<gap>", s)
    s = _XTOK_RE.sub("<gap>", s)
    s = s.translate(_CHAR_TRANS).replace("ₓ", "")
    s = _WS_RE.sub(" ", s).strip()
    return s

# ============================================================
# Inference
# ============================================================
def run_model(model_path, input_texts, device, model_name="model"):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"\n{'='*60}")
    print(f"Loading {model_name}: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    model = model.to(device).eval()

    if device.type == "cuda":
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    translations = []
    start = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(input_texts), BATCH_SIZE):
            batch = input_texts[i:i+BATCH_SIZE]
            tok = tokenizer(batch, max_length=384, padding=True, truncation=True, return_tensors="pt")
            out = model.generate(
                input_ids=tok.input_ids.to(device),
                attention_mask=tok.attention_mask.to(device),
                num_beams=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
                repetition_penalty=1.2,
                use_cache=True,
            )
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            translations.extend([d.strip() for d in dec])

            if (i // BATCH_SIZE) % 10 == 0:
                done = min(i + BATCH_SIZE, len(input_texts))
                print(f"  [{done}/{len(input_texts)}]", end="", flush=True)

    elapsed = time.time() - start
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/len(input_texts):.2f}s/sample)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak:.1f} GB")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return translations

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load eval data from train.csv
    # NOTE: models likely trained on this data, absolute scores are inflated
    # Only the relative comparison between models matters
    print(f"\nLoading: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total rows: {len(df)}")
    src_col, tgt_col = "transliteration", "translation"

    # Filter out empty rows
    df = df.dropna(subset=[src_col, tgt_col])
    df = df[df[tgt_col].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"After filtering: {len(df)}")

    # Sample
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(df), min(NUM_EVAL_SAMPLES, len(df)), replace=False)
    sample_df = df.iloc[sample_idx].reset_index(drop=True)
    print(f"Eval samples: {len(sample_df)}")

    # Preprocess
    sources = [preprocess(str(t)) for t in sample_df[src_col].tolist()]
    references = [str(t).strip() for t in sample_df[tgt_col].tolist()]
    input_texts = ["translate Akkadian to English: " + s for s in sources]

    # Run models
    results = {}

    # XL model
    if os.path.exists(XL_MODEL_PATH):
        xl_trans = run_model(XL_MODEL_PATH, input_texts, device, "ByT5-XL (4B)")
        results["ByT5-XL (4B)"] = xl_trans
    else:
        print(f"XL model not found at {XL_MODEL_PATH}")

    # Large model (if available locally)
    if LARGE_MODEL_PATH and os.path.exists(LARGE_MODEL_PATH):
        large_trans = run_model(LARGE_MODEL_PATH, input_texts, device, "ByT5-large (mbr-v2)")
        results["ByT5-large (mbr-v2)"] = large_trans

    # Evaluate
    print(f"\n{'='*60}")
    print(f"EVALUATION ({len(sample_df)} samples)")
    print(f"{'='*60}")

    for name, translations in results.items():
        bleus, chrfs, geos = [], [], []
        for hyp, ref in zip(translations, references):
            b = bleu_score(hyp, ref)
            c = chrfpp_score(hyp, ref)
            g = geomean(b, c)
            bleus.append(b)
            chrfs.append(c)
            geos.append(g)

        print(f"\n--- {name} ---")
        print(f"  BLEU:      {np.mean(bleus):.2f}")
        print(f"  chrF++:    {np.mean(chrfs):.2f}")
        print(f"  GeoMean:   {np.mean(geos):.2f}")
        print(f"  Empty:     {sum(1 for t in translations if not t.strip())}")

        # Show some examples
        print(f"\n  Sample translations:")
        for idx in [0, len(translations)//3, 2*len(translations)//3]:
            if idx < len(translations):
                hyp_preview = translations[idx][:100] + "..." if len(translations[idx]) > 100 else translations[idx]
                ref_preview = references[idx][:100] + "..." if len(references[idx]) > 100 else references[idx]
                b = bleu_score(translations[idx], references[idx])
                c = chrfpp_score(translations[idx], references[idx])
                print(f"  [{idx}] GeoMean={geomean(b,c):.1f}")
                print(f"    Hyp: {hyp_preview}")
                print(f"    Ref: {ref_preview}")

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'BLEU':>8} {'chrF++':>8} {'GeoMean':>8}")
        print("-" * 51)
        for name, translations in results.items():
            bleus = [bleu_score(h, r) for h, r in zip(translations, references)]
            chrfs = [chrfpp_score(h, r) for h, r in zip(translations, references)]
            geos = [geomean(b, c) for b, c in zip(bleus, chrfs)]
            print(f"{name:<25} {np.mean(bleus):>8.2f} {np.mean(chrfs):>8.2f} {np.mean(geos):>8.2f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
