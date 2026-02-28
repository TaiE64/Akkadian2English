#!/usr/bin/env python3
"""
Prepare training data for QLoRA fine-tuning.
Loads enriched + ORACC data, applies format rules, splits train/val.

Usage:
    conda activate kaggle_deep_past
    python qlora/prepare_data.py
"""

import os
import re
import sys
import math
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["PYTHONIOENCODING"] = "utf-8"

# ============================================================
# Preprocessing helpers (from notebook OptimizedPreprocessor)
# ============================================================

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

TRANSLIT_SPECIAL_CHAR_MAP = {
    "ḫ": "h", "Ḫ": "H",
    "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
}
TRANSLIT_SPECIAL_SEQ_MAP = {"mₓ": "m", "zₓ": "z"}
_SUB_X = "ₓ"
_CHAR_TRANS = str.maketrans(TRANSLIT_SPECIAL_CHAR_MAP)
_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")
_WS_RE = re.compile(r"\s+")

# Gaps
_TAG_GAP_RE      = re.compile(r"<\s*gap\s*>", re.I)
_TAG_BIGGAP_RE   = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_BARE_BIGGAP_RE  = re.compile(r"\bbig[\s_\-]*gap\b", re.I)
_ELLIPSIS_RE     = re.compile(r"(?:\.{3,}|…+|……|\[\.+\])")
_BRACKET_X_RE    = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_XTOKEN_RUN_RE   = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE         = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE         = re.compile(r"(?<!\w)x(?!\w)", re.I)

# Float artifacts (V15: Unicode fractions, 4-digit threshold)
_ALLOWED_FRACS = [
    (1.0/6.0, "⅙"), (1.0/4.0, "¼"), (1.0/3.0, "⅓"),
    (1.0/2.0, "½"), (5.0/8.0, "⅝"), (2.0/3.0, "⅔"),
    (3.0/4.0, "¾"), (5.0/6.0, "⅚"),
]
_FRAC_TOL = 2e-3
_FLOAT_ARTIFACT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")  # V15: 4+ digits

# Translation patterns
_PN_RE = re.compile(r"\bPN\b")
_QUOTES_RE = re.compile(r'["""'']')
_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_MONTH_ROMAN_RE = re.compile(
    r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE
)
_ROMAN2INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}
# V15: keep () and quotes, only strip dashes/brackets/diacritics
_FORBIDDEN_CHARS = '—–<>⌈⌋⌊+ʾ'
_FORBIDDEN_TRANS = str.maketrans("", "", _FORBIDDEN_CHARS)

# (break), (large break), (n broken lines) -> <gap>  (V15)
_BREAK_RE = re.compile(r"\(\s*(?:break|broken|large break|\d+\s+broken lines?|rest\s+broken)\s*\)", re.I)

# Stray marks (V15)
_STRAY_ANGLE_RE = re.compile(r'<<\s*>>')
_STRAY_SINGLE_ANGLE_RE = re.compile(r'(?<!gap)(?<!<)<(?!gap)(?!<)\s*>')

# Grammar annotations (V15 style, standalone)
_GRAM_FEM_RE = re.compile(r'(?<!\w)fem\.\s*')
_GRAM_SING_RE = re.compile(r'(?<!\w)sing\.\s*')
_GRAM_PL_RE = re.compile(r'(?<!\w)pl\.\s*')
_GRAM_PLURAL_RE = re.compile(r'(?<!\w)plural(?!\w)\s*')
_GRAM_QUERY_RE = re.compile(r'\(\?\)')


def ascii_to_diacritics(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


def normalize_gaps(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _TAG_BIGGAP_RE.sub("<gap>", t)
    t = _TAG_GAP_RE.sub("<gap>", t)
    t = _BARE_BIGGAP_RE.sub("<gap>", t)
    t = _BREAK_RE.sub("<gap>", t)  # V15: (break), (large break), etc.
    t = _XTOKEN_RUN_RE.sub("<gap>", t)
    t = _ELLIPSIS_RE.sub("<gap>", t)
    t = _BRACKET_X_RE.sub("<gap>", t)
    t = _XRUN_RE.sub("<gap>", t)
    t = _XTOK_RE.sub("<gap>", t)
    return t


def collapse_gaps(text: str) -> str:
    tokens = str(text).split()
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "<gap>":
            j = i
            while j < len(tokens) and tokens[j] == "<gap>":
                j += 1
            out.append("<gap>")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _canon_decimal_str(x: float) -> str:
    # V15: simple truncation to 4 decimal places
    return f"{x:.4f}".rstrip("0").rstrip(".")


def normalize_float_artifacts(text: str) -> str:
    s = "" if text is None else str(text)
    def repl(m):
        try:
            return _canon_decimal_str(float(m.group(1)))
        except Exception:
            return m.group(0)
    return _FLOAT_ARTIFACT_RE.sub(repl, s)


# ============================================================
# Source (transliteration) preprocessing
# ============================================================
def preprocess_source(text: str) -> str:
    """Apply data_format_rules.md to transliteration."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = ascii_to_diacritics(s)
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    s = normalize_gaps(s)
    s = s.replace('KÙ.B.', 'KÙ.BABBAR')  # V15: expand abbreviation
    for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
        s = s.replace(k, v)
    s = s.translate(_CHAR_TRANS).replace(_SUB_X, "")
    s = normalize_float_artifacts(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# Target (translation) preprocessing
# ============================================================
def preprocess_target(text: str) -> str:
    """Apply data_format_rules.md to translation (aligned with V15)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # Normalize gaps (includes break patterns)
    s = normalize_gaps(s)
    # PN -> <gap>
    s = _PN_RE.sub("<gap>", s)
    # Collapse consecutive gaps
    s = collapse_gaps(s)

    # Remove grammatical-tag parentheses (V15: use _SOFT_GRAM_PARENS_RE)
    s = _SOFT_GRAM_PARENS_RE.sub(" ", s)

    # V15: also remove standalone grammar annotations
    s = _GRAM_FEM_RE.sub("", s)
    s = _GRAM_SING_RE.sub("", s)
    s = _GRAM_PL_RE.sub("", s)
    s = _GRAM_PLURAL_RE.sub("", s)
    s = _GRAM_QUERY_RE.sub("", s)

    # V15: do NOT remove quotes — keep "" '' and apostrophe
    # (commented out: s = _QUOTES_RE.sub("", s))

    # V15: Determinative replacement in target
    s = re.sub(r'\(d\)', '{d}', s)
    s = re.sub(r'\(ki\)', '{ki}', s)
    s = re.sub(r'\(TÚG\)', 'TÚG', s)

    # V15: Remove stray marks << >> and < > (but not <gap>)
    s = _STRAY_ANGLE_RE.sub("", s)
    s = _STRAY_SINGLE_ANGLE_RE.sub("", s)

    # Protect <gap>, remove forbidden chars (V15: keeps () and quotes), restore
    s = s.replace("<gap>", "\x00GAP\x00")
    s = s.translate(_FORBIDDEN_TRANS)
    s = s.replace("\x00GAP\x00", " <gap> ")

    # Float artifacts (V15: 4-digit truncation)
    s = normalize_float_artifacts(s)

    # Month roman -> int
    def month_repl(m):
        r = m.group(1).upper()
        return f"Month {_ROMAN2INT.get(r, r)}"
    s = _MONTH_ROMAN_RE.sub(month_repl, s)

    # Clean whitespace
    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# Data loading and merging
# ============================================================
def load_and_merge_data(cfg):
    """Load all data sources, deduplicate, return unified DataFrame."""
    print("Loading data sources...")

    # Load enriched data (highest quality)
    enriched = pd.read_csv(cfg.enriched_path, encoding="utf-8")
    enriched = enriched.rename(columns={"transliteration": "src", "translation": "tgt"})
    enriched["is_oa"] = True  # Old Assyrian domain
    print(f"  Enriched: {len(enriched)} rows")

    # Load ORACC data (additional, multi-period)
    oracc = pd.read_csv(cfg.oracc_path, encoding="utf-8")
    oracc_extra = oracc[oracc["origin"] != "competition"].copy()
    oracc_extra = oracc_extra.rename(columns={"source": "src", "target": "tgt"})
    oracc_extra["is_oa"] = False  # Non-OA data
    print(f"  ORACC (non-competition): {len(oracc_extra)} rows")

    # Combine
    combined = pd.concat([
        enriched[["src", "tgt", "is_oa"]],
        oracc_extra[["src", "tgt", "is_oa"]],
    ], ignore_index=True)

    # Drop rows with empty/NaN source or target
    combined = combined.dropna(subset=["src", "tgt"])
    combined = combined[combined["src"].str.strip() != ""]
    combined = combined[combined["tgt"].str.strip() != ""]
    print(f"  After cleaning: {len(combined)} rows")

    # Deduplicate by source text (keep first)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["src"], keep="first").reset_index(drop=True)
    print(f"  After dedup: {len(combined)} rows (removed {before_dedup - len(combined)})")

    return combined


def prepare_dataset(cfg):
    """Full data preparation pipeline."""
    combined = load_and_merge_data(cfg)

    # Apply preprocessing
    print("\nApplying preprocessing...")
    combined["src_processed"] = combined["src"].apply(preprocess_source)
    combined["tgt_processed"] = combined["tgt"].apply(preprocess_target)

    # Add task prefix
    combined["input_text"] = cfg.task_prefix + combined["src_processed"]

    # Filter out very short entries
    mask = (combined["src_processed"].str.len() > 5) & (combined["tgt_processed"].str.len() > 3)
    combined = combined[mask].reset_index(drop=True)
    print(f"After short filter: {len(combined)} rows")

    # Filter out entries too long to be useful (truncation > 50% wastes compute)
    src_len = combined["input_text"].str.len()  # includes task prefix
    tgt_len = combined["tgt_processed"].str.len()
    long_mask = (src_len <= cfg.max_source_len * 2) & (tgt_len <= cfg.max_target_len * 2)
    n_removed = (~long_mask).sum()
    combined = combined[long_mask].reset_index(drop=True)
    print(f"After long filter: {len(combined)} rows (removed {n_removed} too-long entries)")

    # Train/val split BEFORE upsampling (prevent data leakage)
    train_df, val_df = train_test_split(
        combined, test_size=cfg.val_ratio, random_state=cfg.seed
    )
    print(f"\nSplit (before upsample): Train={len(train_df)}, Val={len(val_df)}")

    # Upsample OA data in TRAIN only
    train_oa = train_df[train_df["is_oa"]]
    train_non_oa = train_df[~train_df["is_oa"]]
    print(f"  Train OA: {len(train_oa)} rows (will upsample {cfg.oa_upsample}x)")
    print(f"  Train Non-OA: {len(train_non_oa)} rows")

    train_df = pd.concat([train_oa] * cfg.oa_upsample + [train_non_oa], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    val_df = val_df.reset_index(drop=True)
    print(f"\nFinal: Train={len(train_df)}, Val={len(val_df)}")

    # Save
    out_dir = Path(cfg.prepared_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    print(f"\nSaved to {out_dir}/")

    # Print samples
    print("\n" + "=" * 60)
    print("Sample preprocessed data:")
    print("=" * 60)
    for i in range(min(3, len(val_df))):
        row = val_df.iloc[i]
        print(f"\n  Source (raw):  {str(row['src'])[:80]}...")
        print(f"  Source (proc): {str(row['src_processed'])[:80]}...")
        print(f"  Target (raw):  {str(row['tgt'])[:80]}...")
        print(f"  Target (proc): {str(row['tgt_processed'])[:80]}...")

    # Statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    src_lens = train_df["input_text"].str.len()
    tgt_lens = train_df["tgt_processed"].str.len()
    print(f"  Source length (bytes): mean={src_lens.mean():.0f}, "
          f"median={src_lens.median():.0f}, max={src_lens.max()}")
    print(f"  Target length (bytes): mean={tgt_lens.mean():.0f}, "
          f"median={tgt_lens.median():.0f}, max={tgt_lens.max()}")
    print(f"  Source > {cfg.max_source_len}: {(src_lens > cfg.max_source_len).sum()} "
          f"({(src_lens > cfg.max_source_len).mean()*100:.1f}%)")
    print(f"  Target > {cfg.max_target_len}: {(tgt_lens > cfg.max_target_len).sum()} "
          f"({(tgt_lens > cfg.max_target_len).mean()*100:.1f}%)")

    return train_df, val_df


if __name__ == "__main__":
    # Change to project root
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import QLoRAConfig

    cfg = QLoRAConfig()
    prepare_dataset(cfg)
