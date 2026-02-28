"""
Test: what preprocessing does Venom's model actually expect?
Compare different preprocessing levels on the same samples.
"""
import torch
import os
import re
import math
import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# Config
# ============================================================
XL_MODEL_PATH = r"C:\Users\29421\Desktop\kaggle_Challenge\models\byt5-xl-akkadian"
TRAIN_CSV = r"C:\Users\29421\Desktop\kaggle_Challenge\data\competition\train.csv"
NUM_EVAL_SAMPLES = 50
SEED = 42
MAX_NEW_TOKENS = 256
MAX_LENGTH = 384

# ============================================================
# Metrics
# ============================================================
def _extract_char_ngrams(s, n):
    return Counter([s[i:i+n] for i in range(len(s) - n + 1)])

def _extract_word_ngrams(s, n):
    words = s.split()
    if len(words) < n:
        return Counter()
    return Counter([tuple(words[i:i+n]) for i in range(len(words) - n + 1)])

def chrfpp_score(hypothesis, reference, char_order=6, word_order=2, beta=2.0):
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

def bleu_score_smooth(hypothesis, reference, max_n=4):
    hyp_words = hypothesis.strip().split()
    ref_words = reference.strip().split()
    if not hyp_words or not ref_words:
        return 0.0
    bp = min(1.0, np.exp(1 - len(ref_words) / max(len(hyp_words), 1)))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = Counter([tuple(hyp_words[i:i+n]) for i in range(len(hyp_words) - n + 1)])
        ref_ng = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1)])
        common = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        prec = (common + 1) / (total + 1)
        log_avg += np.log(prec) / max_n
    return bp * np.exp(log_avg) * 100.0

def geomean(bleu, chrf):
    if bleu <= 0 or chrf <= 0:
        return 0.0
    return math.sqrt(bleu * chrf)

# ============================================================
# Different preprocessing levels
# ============================================================

# --- Level 0: Raw (no preprocessing at all) ---
def preprocess_raw(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    return str(text).strip()

# --- Level 1: Minimal (just whitespace cleanup) ---
def preprocess_minimal(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Level 2: Our current full preprocessing ---
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})
_CHAR_MAP = {"ḫ":"h","Ḫ":"H","ʾ":"","₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
             "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","—":"-","–":"-"}
_CHAR_TRANS = str.maketrans(_CHAR_MAP)
_WS_RE = re.compile(r"\s+")
_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")

def preprocess_full(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    s = s.replace("sz","š").replace("SZ","Š").replace("s,","ṣ").replace("S,","Ṣ").replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    s = s.translate(_CHAR_TRANS).replace("ₓ", "")
    s = _WS_RE.sub(" ", s).strip()
    return s

# --- Level 3: Keep diacritics (don't convert ḫ→h, keep subscripts) ---
def preprocess_keep_diacritics(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    # Only do ASCII→diacritics conversion, but keep ḫ as ḫ
    s = s.replace("sz","š").replace("SZ","Š").replace("s,","ṣ").replace("S,","Ṣ").replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    # DON'T convert ḫ→h, DON'T convert subscripts
    s = _WS_RE.sub(" ", s).strip()
    return s

# --- Level 4: Gap normalization added ---
_TAG_GAP_RE = re.compile(r"<\s*gap\s*>", re.I)
_TAG_BIGGAP_RE = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_BARE_BIGGAP_RE = re.compile(r"\bbig[\s_\-]*gap\b", re.I)
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)")
_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)

def preprocess_full_gaps(text):
    """Full preprocessing + gap normalization."""
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    s = s.replace("sz","š").replace("SZ","Š").replace("s,","ṣ").replace("S,","Ṣ").replace("t,","ṭ").replace("T,","Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    s = s.translate(_CHAR_TRANS).replace("ₓ", "")
    # Gap normalization
    s = _TAG_BIGGAP_RE.sub("<gap>", s)
    s = _TAG_GAP_RE.sub("<gap>", s)
    s = _BARE_BIGGAP_RE.sub("<gap>", s)
    s = _XTOKEN_RUN_RE.sub("<gap>", s)
    s = _ELLIPSIS_RE.sub("<gap>", s)
    s = _BRACKET_X_RE.sub("<gap>", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=["transliteration", "translation"])
    df = df[df["translation"].str.strip().str.len() > 0].reset_index(drop=True)
    np.random.seed(SEED)
    idx = np.random.choice(len(df), min(NUM_EVAL_SAMPLES, len(df)), replace=False)
    sample_df = df.iloc[idx].reset_index(drop=True)

    raw_sources = sample_df["transliteration"].tolist()
    references = [str(t).strip() for t in sample_df["translation"].tolist()]
    print(f"Eval samples: {len(sample_df)}")

    # Show a few examples of different preprocessing
    print("\n--- Preprocessing comparison (first 3 samples) ---")
    for i in range(min(3, len(raw_sources))):
        r = raw_sources[i]
        print(f"\n[{i}] Raw:            {str(r)[:100]}")
        print(f"    Minimal:         {preprocess_minimal(r)[:100]}")
        print(f"    Full:            {preprocess_full(r)[:100]}")
        print(f"    Keep diacritics: {preprocess_keep_diacritics(r)[:100]}")
        print(f"    Full+gaps:       {preprocess_full_gaps(r)[:100]}")

    # Define preprocessing configs
    configs = {
        "raw":             preprocess_raw,
        "minimal":         preprocess_minimal,
        "full (current)":  preprocess_full,
        "keep_diacritics": preprocess_keep_diacritics,
        "full+gaps":       preprocess_full_gaps,
    }

    # Load model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print(f"\nLoading ByT5-XL...")
    tokenizer = AutoTokenizer.from_pretrained(XL_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(XL_MODEL_PATH, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"Loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    amp_dtype = torch.bfloat16 if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float16

    # Run each config
    results = {}
    for name, preproc_fn in configs.items():
        print(f"\n--- Testing: {name} ---")
        sources = [preproc_fn(t) for t in raw_sources]
        input_texts = ["translate Akkadian to English: " + s for s in sources]

        translations = []
        with torch.inference_mode():
            for i, text in enumerate(input_texts):
                tok = tokenizer([text], max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
                input_ids = tok.input_ids.to(device)
                attn = tok.attention_mask.to(device)

                ctx = torch.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
                with ctx:
                    out = model.generate(
                        input_ids=input_ids, attention_mask=attn,
                        do_sample=False, num_beams=4,
                        max_new_tokens=MAX_NEW_TOKENS, length_penalty=1.3,
                        early_stopping=True, use_cache=True,
                    )
                dec = tokenizer.decode(out[0], skip_special_tokens=True).strip()
                translations.append(dec)

                if (i+1) % 10 == 0:
                    print(f"  [{i+1}/{len(input_texts)}]", end="", flush=True)
                if i % 15 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
        print()

        bleus = [bleu_score_smooth(h, r) for h, r in zip(translations, references)]
        chrfs = [chrfpp_score(h, r) for h, r in zip(translations, references)]
        geos = [geomean(b, c) for b, c in zip(bleus, chrfs)]
        results[name] = {
            "translations": translations,
            "bleu": np.mean(bleus),
            "chrf": np.mean(chrfs),
            "geomean": np.mean(geos),
        }

    # Print results
    print(f"\n{'='*65}")
    print(f"RESULTS ({len(sample_df)} samples)")
    print(f"{'='*65}")
    print(f"{'Preprocessing':<20} {'BLEU':>8} {'chrF++':>8} {'GeoMean':>8}")
    print("-" * 46)
    for name, r in results.items():
        print(f"{name:<20} {r['bleu']:>8.2f} {r['chrf']:>8.2f} {r['geomean']:>8.2f}")

    # Show cases where raw vs full differ
    print(f"\n--- Cases where raw != full output ---")
    raw_trans = results["raw"]["translations"]
    full_trans = results["full (current)"]["translations"]
    diff_count = 0
    for i, (rt, ft, ref) in enumerate(zip(raw_trans, full_trans, references)):
        if rt != ft:
            diff_count += 1
            if diff_count <= 5:
                g_raw = geomean(bleu_score_smooth(rt, ref), chrfpp_score(rt, ref))
                g_full = geomean(bleu_score_smooth(ft, ref), chrfpp_score(ft, ref))
                print(f"\n  [{i}] raw GeoMean={g_raw:.1f}, full GeoMean={g_full:.1f}")
                print(f"    Raw output:  {rt[:100]}")
                print(f"    Full output: {ft[:100]}")
                print(f"    Reference:   {ref[:100]}")
    print(f"\n  Total different: {diff_count}/{len(raw_trans)}")

    print("\nDone!")

if __name__ == "__main__":
    main()
