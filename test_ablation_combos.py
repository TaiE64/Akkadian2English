#!/usr/bin/env python3
"""
Ablation test: pre/post processing combinations for ByT5-XL.

Runs model inference ONCE per preprocessing config, then applies
multiple postprocessing configs to cached raw outputs.

Usage:
    conda activate kaggle_deep_past
    python test_ablation_combos.py --n_samples 200
    python test_ablation_combos.py --n_samples 200 --skip_pre_v11   # only run pre_v15
"""

import argparse
import os
import re
import math
import time
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════
try:
    import sacrebleu

    def compute_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
        geo = math.sqrt(max(bleu.score, 0) * max(chrf.score, 0))
        return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2), "geomean": round(geo, 2)}

except ImportError:
    print("WARNING: sacrebleu not found, installing...")
    os.system("pip install sacrebleu")
    import sacrebleu

    def compute_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
        geo = math.sqrt(max(bleu.score, 0) * max(chrf.score, 0))
        return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2), "geomean": round(geo, 2)}


# ═══════════════════════════════════════════════════════════════
# Shared regex patterns
# ═══════════════════════════════════════════════════════════════
_WS_RE = re.compile(r"\s+")

# -- Gaps --
_TAG_GAP_RE = re.compile(r"<\s*gap\s*>", re.I)
_TAG_BIGGAP_RE = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_BARE_BIGGAP_RE = re.compile(r"\bbig[\s_\-]*gap\b", re.I)
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+|……|\[\.+\])")
_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE = re.compile(r"(?<!\w)x(?!\w)", re.I)
_BREAK_RE = re.compile(
    r"\(\s*(?:break|broken|large break|\d+\s+broken lines?|rest\s+broken)\s*\)", re.I
)

# -- Diacritics --
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans(
    {"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"}
)
_GRAVE = str.maketrans(
    {"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"}
)

TRANSLIT_SPECIAL_CHAR_MAP = {
    "ḫ": "h", "Ḫ": "H", "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
}
TRANSLIT_SPECIAL_SEQ_MAP = {"mₓ": "m", "zₓ": "z"}
_CHAR_TRANS = str.maketrans(TRANSLIT_SPECIAL_CHAR_MAP)
_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)")
_PN_RE = re.compile(r"\bPN\b")
_QUOTES_RE = re.compile(r'[\u201c\u201d\u201e\u201f\u2018\u2019\u201a\u201b]')
_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)

# -- Month roman --
_MONTH_ROMAN_RE = re.compile(
    r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE
)
_ROMAN2INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}

# -- Fraction map (for conversion) --
FRAC_MAP = {
    "0.5": "½", "0.25": "¼", "0.75": "¾",
    "0.3333": "⅓", "0.6666": "⅔",
    "0.1666": "⅙", "0.8333": "⅚",
    "0.625": "⅝",
}

# -- V11's fraction-convert map (decimal strings) --
_V11_ALLOWED_FRACS = [
    (1.0 / 6.0, "0.16666"), (1.0 / 4.0, "0.25"), (1.0 / 3.0, "0.33333"),
    (1.0 / 2.0, "0.5"), (2.0 / 3.0, "0.66666"), (3.0 / 4.0, "0.75"),
    (5.0 / 6.0, "0.83333"),
]
_FRAC_TOL = 2e-3


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════
def normalize_gaps(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _TAG_BIGGAP_RE.sub("<gap>", t)
    t = _TAG_GAP_RE.sub("<gap>", t)
    t = _BARE_BIGGAP_RE.sub("<gap>", t)
    t = _XTOKEN_RUN_RE.sub("<gap>", t)
    t = _ELLIPSIS_RE.sub("<gap>", t)
    t = _BRACKET_X_RE.sub("<gap>", t)
    t = _XRUN_RE.sub("<gap>", t)
    t = _XTOK_RE.sub("<gap>", t)
    t = _BREAK_RE.sub("<gap>", t)
    return t


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


def collapse_gap_runs(text: str) -> str:
    tokens = str(text).split()
    out, i, n = [], 0, len(tokens)
    while i < n:
        if tokens[i] == "<gap>":
            j = i
            while j < n and tokens[j] == "<gap>":
                j += 1
            out.append("<gap>")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _month_repl(m):
    r = m.group(1).upper()
    return f"Month {_ROMAN2INT.get(r, r)}"


# ═══════════════════════════════════════════════════════════════
# Preprocessor (configurable)
# ═══════════════════════════════════════════════════════════════
class Preprocessor:
    """
    Configurable preprocessor for transliteration input.

    Config flags:
      - ku_babbar: Convert KÙ.B. → KÙ.BABBAR (V15=True, V11=False)
      - float_threshold: Min digits after decimal to trigger normalization (V15=4, V11=6)
      - float_truncate_only: Just truncate (V15=True) vs fraction-convert (V11=False)
    """

    def __init__(self, ku_babbar=True, float_threshold=4, float_truncate_only=True):
        self.ku_babbar = ku_babbar
        self.float_threshold = float_threshold
        self.float_truncate_only = float_truncate_only
        self._float_re = re.compile(
            rf"(?<![\w/])(\d+\.\d{{{float_threshold},}})(?![\w/])"
        )

    def _canon_float(self, x: float) -> str:
        if self.float_truncate_only:
            return f"{x:.4f}".rstrip("0").rstrip(".")
        # V11 style: try to match known fractions
        ip = int(math.floor(x + 1e-12))
        frac = x - ip
        best = None
        for v, dec in _V11_ALLOWED_FRACS:
            d = abs(frac - v)
            if best is None or d < best[0]:
                best = (d, dec)
        if best and best[0] <= _FRAC_TOL:
            dec = best[1]
            if ip == 0:
                return dec
            return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
        return f"{x:.5f}".rstrip("0").rstrip(".")

    def preprocess(self, text: str) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        s = str(text)
        s = ascii_to_diacritics(s)
        s = _DET_PARENS_RE.sub(r"{\1}", s)
        s = normalize_gaps(s)
        for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
            s = s.replace(k, v)
        s = s.translate(_CHAR_TRANS).replace("ₓ", "")
        s = self._float_re.sub(lambda m: self._canon_float(float(m.group(1))), s)
        if self.ku_babbar:
            s = s.replace("KÙ.B.", "KÙ.BABBAR")
        s = _WS_RE.sub(" ", s).strip()
        return s

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]


# ═══════════════════════════════════════════════════════════════
# Postprocessor (configurable)
# ═══════════════════════════════════════════════════════════════
class Postprocessor:
    """
    Configurable postprocessor for model translations.

    Toggle flags (all bool, default matches V15 current):
      - remove_quotes: Remove curly/ASCII quotes (V11=True, V15=False)
      - remove_parens: Include () in forbidden chars (V11=True, V15=False)
      - determinatives: (d)→{d}, (ki)→{ki}, (TÚG)→TÚG (V15=True, V11=False)
      - grammar_removal: Remove fem./sing./pl. etc. (V15=True, V11=False)
      - stray_marks: Remove <<>>, < > (V15=True, V11=False)
      - fraction_conversion: 0.5→½ etc. (disabled in both V11 and V15)
      - word_replacements: -gold→pašallum gold etc. (disabled in both)
      - shekel_conversions: shekel→grains (disabled in both)
      - slash_handling: "word / word" → keep first (disabled in both)
      - float_4digit: Use 4-digit threshold (V15=True, V11=False=6-digit)
      - float_truncate_only: Just truncate (V15=True, V11=False=fraction-convert)
      - fix_repeats: Remove repeated words (both=True)
    """

    def __init__(self, **kwargs):
        self.remove_quotes = kwargs.get("remove_quotes", False)
        self.remove_parens = kwargs.get("remove_parens", False)
        self.determinatives = kwargs.get("determinatives", True)
        self.grammar_removal = kwargs.get("grammar_removal", True)
        self.stray_marks = kwargs.get("stray_marks", True)
        self.fraction_conversion = kwargs.get("fraction_conversion", False)
        self.word_replacements = kwargs.get("word_replacements", False)
        self.shekel_conversions = kwargs.get("shekel_conversions", False)
        self.slash_handling = kwargs.get("slash_handling", False)
        self.float_4digit = kwargs.get("float_4digit", True)
        self.float_truncate_only = kwargs.get("float_truncate_only", True)
        self.fix_repeats = kwargs.get("fix_repeats", True)

        # Build forbidden chars
        base_forbidden = "—–<>⌈⌋⌊+ʾ"
        if self.remove_parens:
            base_forbidden += "()"
        self.forbidden_trans = str.maketrans("", "", base_forbidden)

        # Float artifact regex
        threshold = 4 if self.float_4digit else 6
        self._float_re = re.compile(rf"(?<![\w/])(\d+\.\d{{{threshold},}})(?![\w/])")

        # Patterns
        self._gap_legacy = re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I)
        self._big_gap_legacy = re.compile(r"(\.{3,}|…|\[\.+\])")
        self._repeated_words = re.compile(r"\b(\w+)(?:\s+\1\b)+")
        self._punct_space = re.compile(r"\s+([.,:;])")
        self._repeated_punct = re.compile(r"([.,:;])\1+")

        # Slash pattern: "word / word" → keep first word
        self._slash_re = re.compile(r"(\S+)\s*/\s*(\S+)")

        # Shekel patterns (most specific first)
        self._shekel_patterns = [
            (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
            (re.compile(r"7\s*/\s*12\s+shekels?", re.I), "½ shekel 15 grains"),
            (re.compile(r"5\s*/\s*12\s+shekels?", re.I), "15 grains"),
            (re.compile(r"1\s*/\s*12\s+\(?\s*shekels?\s*\)?", re.I), "⅔ shekel 15 grains"),
        ]

        # Word replacement patterns
        self._word_replace_patterns = [
            (re.compile(r"(?<!\w)-gold(?!\w)", re.I), "pašallum gold"),
            (re.compile(r"(?<!\w)-tax(?!\w)", re.I), "šadduātum tax"),
            (re.compile(r"(?<!\w)textiles(?!\w)", re.I), "kutānum textiles"),
        ]

        # Fraction conversion patterns (0.5 → ½ etc.)
        self._frac_patterns = []
        for dec_str, uni_frac in FRAC_MAP.items():
            # Match standalone or as part of "N.xxxx"
            pattern = re.compile(r"(?<!\d)" + re.escape(dec_str) + r"(?!\d)")
            self._frac_patterns.append((pattern, uni_frac))

    def _canon_float(self, x: float) -> str:
        if self.float_truncate_only:
            return f"{x:.4f}".rstrip("0").rstrip(".")
        # V11 style
        ip = int(math.floor(x + 1e-12))
        frac = x - ip
        best = None
        for v, dec in _V11_ALLOWED_FRACS:
            d = abs(frac - v)
            if best is None or d < best[0]:
                best = (d, dec)
        if best and best[0] <= _FRAC_TOL:
            dec = best[1]
            if ip == 0:
                return dec
            return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
        return f"{x:.5f}".rstrip("0").rstrip(".")

    def postprocess(self, text: str) -> str:
        if not text or not isinstance(text, str) or not text.strip():
            return ""
        s = text

        # 0) Gaps + PN (always)
        s = normalize_gaps(s)
        s = _PN_RE.sub("<gap>", s)
        s = _WS_RE.sub(" ", s).strip()

        # 1) Legacy gap patterns (always)
        s = self._gap_legacy.sub("<gap>", s)
        s = self._big_gap_legacy.sub("<gap>", s)

        # 2) Soft gram parens removal (always - both V11 and V15 have this)
        s = _SOFT_GRAM_PARENS_RE.sub(" ", s)

        # 3) Quote removal (V11=True, V15=False)
        if self.remove_quotes:
            s = _QUOTES_RE.sub("", s)

        # 4) Gap collapse (always)
        s = collapse_gap_runs(s)

        # 5) Determinatives (V15 only)
        if self.determinatives:
            s = re.sub(r"\(d\)", "{d}", s)
            s = re.sub(r"\(ki\)", "{ki}", s)
            s = re.sub(r"\(TÚG\)", "TÚG", s)

        # 6) Grammar removal (V15 only)
        if self.grammar_removal:
            s = re.sub(r"(?<!\w)fem\.\s*", "", s)
            s = re.sub(r"(?<!\w)sing\.\s*", "", s)
            s = re.sub(r"(?<!\w)pl\.\s*", "", s)
            s = re.sub(r"(?<!\w)plural(?!\w)\s*", "", s)
            s = re.sub(r"\(\?\)", "", s)

        # 7) Stray marks (V15 only)
        if self.stray_marks:
            s = re.sub(r"<<\s*>>", "", s)
            s = re.sub(r"(?<!gap)(?<!<)<(?!gap)(?!<)\s*>", "", s)

        # 8) Protect <gap>, remove forbidden chars, restore
        s = s.replace("<gap>", "\x00GAP\x00")
        s = s.translate(self.forbidden_trans)
        s = s.replace("\x00GAP\x00", " <gap> ")

        # 9) Float artifact normalization
        s = self._float_re.sub(lambda m: self._canon_float(float(m.group(1))), s)

        # 10) Month roman → int (always)
        s = _MONTH_ROMAN_RE.sub(_month_repl, s)

        # 11) Shekel conversions (optional)
        if self.shekel_conversions:
            for pat, repl in self._shekel_patterns:
                s = pat.sub(repl, s)

        # 12) Word replacements (optional)
        if self.word_replacements:
            for pat, repl in self._word_replace_patterns:
                s = pat.sub(repl, s)

        # 13) Slash handling (optional): "word / word" → first word
        if self.slash_handling:
            s = self._slash_re.sub(r"\1", s)

        # 14) Fraction conversion (optional): 0.5 → ½ etc.
        if self.fraction_conversion:
            for pat, repl in self._frac_patterns:
                s = pat.sub(repl, s)

        # 15) Repeat fixing (usually True)
        if self.fix_repeats:
            s = self._repeated_words.sub(r"\1", s)
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                s = re.sub(pattern, r"\1", s)

        # 16) Punct fixes (always)
        s = self._punct_space.sub(r"\1", s)
        s = self._repeated_punct.sub(r"\1", s)

        # 17) Final whitespace
        s = _WS_RE.sub(" ", s).strip()
        return s

    def postprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.postprocess(t) for t in texts]


# ═══════════════════════════════════════════════════════════════
# Named configurations
# ═══════════════════════════════════════════════════════════════

PREPROCESS_CONFIGS = {
    "pre_v11": {
        "ku_babbar": False,
        "float_threshold": 6,
        "float_truncate_only": False,
    },
    "pre_v15": {
        "ku_babbar": True,
        "float_threshold": 4,
        "float_truncate_only": True,
    },
}

# Postprocessing configs
_V11_POST = {
    "remove_quotes": True,
    "remove_parens": True,
    "determinatives": False,
    "grammar_removal": False,
    "stray_marks": False,
    "fraction_conversion": False,
    "word_replacements": False,
    "shekel_conversions": False,
    "slash_handling": False,
    "float_4digit": False,
    "float_truncate_only": False,
    "fix_repeats": True,
}

_V15_POST = {
    "remove_quotes": False,
    "remove_parens": False,
    "determinatives": True,
    "grammar_removal": True,
    "stray_marks": True,
    "fraction_conversion": False,
    "word_replacements": False,
    "shekel_conversions": False,
    "slash_handling": False,
    "float_4digit": True,
    "float_truncate_only": True,
    "fix_repeats": True,
}

POSTPROCESS_CONFIGS = {
    "post_v11": _V11_POST,
    "post_v15": _V15_POST,
    "post_v15+fracs": {**_V15_POST, "fraction_conversion": True},
    "post_v15+words": {**_V15_POST, "word_replacements": True},
    "post_v15+shekels": {**_V15_POST, "shekel_conversions": True},
    "post_v15+slash": {**_V15_POST, "slash_handling": True},
    "post_v15+full": {
        **_V15_POST,
        "fraction_conversion": True,
        "word_replacements": True,
        "shekel_conversions": True,
        "slash_handling": True,
    },
    # Hybrid: V15 base but keep V11's quote/paren removal
    "post_v15+rmquote": {**_V15_POST, "remove_quotes": True},
    "post_v15+rmparen": {**_V15_POST, "remove_parens": True},
    "post_v15+rmqp": {**_V15_POST, "remove_quotes": True, "remove_parens": True},
}


# ═══════════════════════════════════════════════════════════════
# Model inference
# ═══════════════════════════════════════════════════════════════
def load_model(model_path: str, device: str = "cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"Loading model from {model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def run_inference(
    model, tokenizer, texts: List[str], device: str = "cuda",
    num_beams: int = 4, length_penalty: float = 1.3, max_new_tokens: int = 256,
) -> List[str]:
    """Run model inference on a list of prefixed input texts. Returns raw decoded outputs."""
    import torch
    from tqdm import tqdm

    results = []
    with torch.inference_mode():
        for i, text in enumerate(tqdm(texts, desc="Inference")):
            inputs = tokenizer(
                text, max_length=384, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                early_stopping=True,
                use_cache=True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(decoded)

            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ablation test for pre/post processing")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of train samples to use")
    parser.add_argument("--model_path", type=str, default="models/byt5-xl-akkadian", help="Path to model")
    parser.add_argument("--data_path", type=str, default="data/competition/train.csv", help="Path to train.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--skip_pre_v11", action="store_true", help="Skip pre_v11 inference (save time)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default="ablation_results.json", help="Output JSON file")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Loading data from {args.data_path}")
    train_df = pd.read_csv(args.data_path, encoding="utf-8")
    print(f"Total: {len(train_df)} rows")

    # Filter rows that have both transliteration and translation
    valid = train_df.dropna(subset=["transliteration", "translation"])
    valid = valid[valid["translation"].str.strip() != ""]
    print(f"Valid (non-empty): {len(valid)} rows")

    # Sample
    n = min(args.n_samples, len(valid))
    samples = valid.sample(n=n, random_state=args.seed)
    print(f"Sampled: {n} rows (seed={args.seed})")

    transliterations = samples["transliteration"].tolist()
    references = samples["translation"].tolist()

    # ── Load model ───────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, args.device)

    # ── Run ablation ─────────────────────────────────────────
    all_results = []
    raw_cache = {}  # pre_name → list of raw model outputs

    # Determine which preprocessing configs to run
    pre_configs = dict(PREPROCESS_CONFIGS)
    if args.skip_pre_v11:
        pre_configs.pop("pre_v11", None)
        print("\nSkipping pre_v11 (--skip_pre_v11)")

    for pre_name, pre_cfg in pre_configs.items():
        print(f"\n{'='*70}")
        print(f"PREPROCESSING: {pre_name}")
        print(f"  Config: {pre_cfg}")

        # Preprocess
        preprocessor = Preprocessor(**pre_cfg)
        preprocessed = preprocessor.preprocess_batch(transliterations)

        # Add task prefix
        prefixed = ["translate Akkadian to English: " + t for t in preprocessed]

        # Run inference (expensive!)
        print(f"Running inference on {len(prefixed)} samples...")
        t0 = time.time()
        raw_outputs = run_inference(model, tokenizer, prefixed, args.device)
        elapsed = time.time() - t0
        print(f"Inference done in {elapsed:.1f}s ({elapsed/len(prefixed):.2f}s/sample)")

        raw_cache[pre_name] = raw_outputs

        # Score raw (no postprocessing)
        scores_raw = compute_scores(raw_outputs, references)
        print(f"\n  {pre_name} + post_raw: {scores_raw}")
        all_results.append({
            "preprocess": pre_name,
            "postprocess": "post_raw",
            **scores_raw,
        })

        # Apply each postprocessing config
        for post_name, post_cfg in POSTPROCESS_CONFIGS.items():
            postprocessor = Postprocessor(**post_cfg)
            processed = postprocessor.postprocess_batch(list(raw_outputs))
            scores = compute_scores(processed, references)
            print(f"  {pre_name} + {post_name}: {scores}")
            all_results.append({
                "preprocess": pre_name,
                "postprocess": post_name,
                **scores,
            })

    # ── Results summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Samples: {n}, Model: {args.model_path}")
    print()

    # Sort by geomean descending
    all_results.sort(key=lambda x: x["geomean"], reverse=True)

    # Print table
    print(f"{'Rank':<5} {'Preprocess':<12} {'Postprocess':<20} {'BLEU':>7} {'chrF++':>7} {'GeoMean':>8}")
    print("-" * 65)
    for i, r in enumerate(all_results):
        marker = " ***" if i == 0 else ""
        print(
            f"{i+1:<5} {r['preprocess']:<12} {r['postprocess']:<20} "
            f"{r['bleu']:>7.2f} {r['chrf']:>7.2f} {r['geomean']:>8.2f}{marker}"
        )

    # Save results
    import json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # ── Show diff from best ──────────────────────────────────
    if all_results:
        best = all_results[0]["geomean"]
        print(f"\nDiff from best ({best:.2f}):")
        # Find V11+V11 and V15+V15 baselines
        for r in all_results:
            diff = r["geomean"] - best
            tag = ""
            if r["preprocess"] == "pre_v11" and r["postprocess"] == "post_v11":
                tag = " ← V11 baseline (Kaggle 38.1)"
            elif r["preprocess"] == "pre_v15" and r["postprocess"] == "post_v15":
                tag = " ← V15 current"
            print(f"  {r['preprocess']:>12} + {r['postprocess']:<20} {diff:>+7.2f}{tag}")


if __name__ == "__main__":
    main()
