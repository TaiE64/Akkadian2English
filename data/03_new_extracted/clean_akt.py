#!/usr/bin/env python3
"""
清洗 akt_extracted.csv + 修复 akt8_extracted.csv 编码问题
输出:
  - akt_cleaned.csv     (训练数据)
  - akt8_cleaned.csv    (验证数据, 编码修复)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, re
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

# ── 从 prepare_clean_data.py 复用的预处理函数 ────────────────────────
SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_V2 = re.compile(r"([aAeEiIuU])(?:2|\u2082)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|\u2083)")
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

_SRC_TAG_GAP_RE      = re.compile(r"<\s*gap\s*>", re.I)
_SRC_TAG_BIGGAP_RE   = re.compile(r"<\s*big[\s_\-]*gap\s*>", re.I)
_SRC_BARE_BIGGAP_RE  = re.compile(r"\bbig[\s_\-]*gap\b", re.I)
_SRC_ELLIPSIS_RE     = re.compile(r"(?:\.{3,}|…+|\[\.+\])")
_SRC_BRACKET_X_RE    = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_SRC_XTOKEN_RUN_RE   = re.compile(r"\bx(?:\s+x)+\b", re.I)
_SRC_XRUN_RE         = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_SRC_XTOK_RE         = re.compile(r"(?<!\w)x(?!\w)", re.I)
_SRC_BREAK_RE        = re.compile(r"\(\s*(?:break|broken|large break|\d+\s+broken lines?|rest\s+broken)\s*\)", re.I)
_FLOAT_ARTIFACT_RE   = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_WS_RE               = re.compile(r"\s+")

def _canon_decimal_str(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")

def ascii_to_diacritics(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

def normalize_src_gaps(text: str) -> str:
    if text is None: return ""
    t = str(text)
    t = _SRC_BREAK_RE.sub("<gap>", t)
    t = _SRC_TAG_BIGGAP_RE.sub("<gap>", t)
    t = _SRC_TAG_GAP_RE.sub("<gap>", t)
    t = _SRC_BARE_BIGGAP_RE.sub("<gap>", t)
    t = _SRC_XTOKEN_RUN_RE.sub("<gap>", t)
    t = _SRC_ELLIPSIS_RE.sub("<gap>", t)
    t = _SRC_BRACKET_X_RE.sub("<gap>", t)
    t = _SRC_XRUN_RE.sub("<gap>", t)
    t = _SRC_XTOK_RE.sub("<gap>", t)
    return t

def normalize_float_artifacts(text: str) -> str:
    s = "" if text is None else str(text)
    def repl(m):
        try: return _canon_decimal_str(float(m.group(1)))
        except Exception: return m.group(1)
    return _FLOAT_ARTIFACT_RE.sub(repl, s)

# 限定词括号替换
_DET_LIST = ["d","mul","ki","lu2","e2","uru","kur","mi","m",
             "geš","ĝeš","tug2","dub","id2","mušen","na4","kuš","u2"]
_DET_RE = re.compile(
    r"\((?:" + "|".join(re.escape(x) for x in sorted(_DET_LIST, key=len, reverse=True)) + r")\)"
)

# 小数 → Unicode 分数
_FRAC_DECIMAL_MAP = {
    ".5": "½", ".25": "¼", ".75": "¾",
    ".3333": "⅓", ".6666": "⅔",
    ".1666": "⅙", ".8333": "⅚", ".625": "⅝",
}
_FLOAT_TRUNC_RE = re.compile(r"(\d+\.\d{4})\d+")
_FLOAT_FRAC_RE  = re.compile(r"(\d+)(\.(?:5|25|75|3333|6666|1666|8333|625))(?=\D|$)")

def _frac_replacer(m):
    i, d = m.group(1), m.group(2)
    frac = _FRAC_DECIMAL_MAP[d]
    return frac if i == "0" else i + frac

def convert_fracs(text):
    text = _FLOAT_TRUNC_RE.sub(r"\1", text)
    return _FLOAT_FRAC_RE.sub(_frac_replacer, text)

_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b")
_ROMAN_TO_INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,
                 "VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}

def convert_months(text):
    return _MONTH_RE.sub(lambda m: f"Month {_ROMAN_TO_INT[m.group(1)]}", text)

def preprocess_src(text):
    if not isinstance(text, str): return text
    s = str(text)
    s = ascii_to_diacritics(s)
    s = _DET_RE.sub(lambda m: "{" + m.group(0)[1:-1] + "}", s)
    s = s.replace("(TÚG)", "TÚG")
    s = normalize_src_gaps(s)
    s = s.replace('KÙ.B.', 'KÙ.BABBAR')
    for k, v in TRANSLIT_SPECIAL_SEQ_MAP.items():
        s = s.replace(k, v)
    s = s.translate(_CHAR_TRANS)
    s = s.replace(_SUB_X, "")
    s = normalize_float_artifacts(s)
    s = _WS_RE.sub(" ", s).strip()
    return s

# tgt 后处理
_GAP_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+|\[\.+\])")
_GAP_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))", re.I)
_GAP_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)
_GAP_XRUN_RE = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_GAP_XTOK_RE = re.compile(r"(?<!\w)x(?!\w)", re.I)
_GAP_BREAK_RE = re.compile(r"\(\s*(?:break|broken|large break|\d+\s+broken lines?|rest\s+broken)\s*\)", re.I)
_GAP_PN_RE = re.compile(r"\bPN\b")
_GAP_MULTI_RE = re.compile(r"(?:<gap>\s*){2,}")

def normalize_tgt_gaps(text):
    text = _GAP_BREAK_RE.sub("<gap>", text)
    text = _GAP_ELLIPSIS_RE.sub("<gap>", text)
    text = _GAP_XTOKEN_RUN_RE.sub("<gap>", text)
    text = _GAP_BRACKET_X_RE.sub("<gap>", text)
    text = _GAP_XRUN_RE.sub("<gap>", text)
    text = _GAP_XTOK_RE.sub("<gap>", text)
    text = _GAP_PN_RE.sub("<gap>", text)
    text = _GAP_MULTI_RE.sub("<gap> ", text)
    return text

_TGT_FORBIDDEN_CHARS_STR = "—–<>⌈⌋⌊+ʾ"
_TGT_FORBIDDEN_TRANS = str.maketrans("", "", _TGT_FORBIDDEN_CHARS_STR)
_TGT_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I,
)
_TGT_BARE_GRAMMAR_RE = re.compile(r"(?<!\w)fem\.\s*|(?<!\w)sing\.\s*|(?<!\w)pl\.\s*|(?<!\w)plural(?!\w)\s*", re.I)
_TGT_UNCERTAIN_RE = re.compile(r"\(\?\)")
_TGT_DOUBLE_ANGLE_RE = re.compile(r"<<\s*>>")
_TGT_STRAY_ANGLE_RE = re.compile(r"(?<!gap)(?<!<)<(?!gap)(?!<)\s*>")
_TGT_DET_RE = re.compile(
    r"\((?:" + "|".join(re.escape(x) for x in sorted(_DET_LIST, key=len, reverse=True)) + r")\)"
)
_TGT_REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_TGT_PUNCT_SPACE_RE = re.compile(r"\s+([.,:;])")
_TGT_REPEATED_PUNCT_RE = re.compile(r"([.,:;])\1+")

def preprocess_tgt(text):
    if not isinstance(text, str): return text
    text = text.replace("\\n", " ").replace("\\r", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\[([^\]]{1,60})\]", r"\1", text)
    text = normalize_tgt_gaps(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _TGT_SOFT_GRAM_PARENS_RE.sub(" ", text)
    text = text.replace("(TÚG)", "TÚG")
    text = _TGT_DET_RE.sub(lambda m: "{" + m.group(0)[1:-1] + "}", text)
    text = _TGT_BARE_GRAMMAR_RE.sub("", text)
    text = _TGT_UNCERTAIN_RE.sub("", text)
    text = _TGT_DOUBLE_ANGLE_RE.sub("", text)
    text = _TGT_STRAY_ANGLE_RE.sub("", text)
    text = _GAP_MULTI_RE.sub("<gap> ", text)
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(_TGT_FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")
    text = normalize_float_artifacts(text)
    text = convert_fracs(text)
    text = convert_months(text)
    text = _TGT_REPEATED_WORD_RE.sub(r"\1", text)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pattern, r"\1", text)
    text = _TGT_PUNCT_SPACE_RE.sub(r"\1", text)
    text = _TGT_REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 噪声检测
NOISE_RE = re.compile(r"[ܐ-ܿ\x00-\x08\x0b-\x1f\x7f]|[ΩΣΘΛΦΨωβ]|[®©™°±×÷√∞]")
META_RE = re.compile(
    r"(?:obverse|reverse|edge|too broken|not translat|untranslat|"
    r"uninscribed|see translation|cf\.|ibid)",
    re.I
)

def is_noisy(text):
    return bool(NOISE_RE.search(str(text)))

def is_meta(text):
    return bool(META_RE.search(str(text)))


# ════════════════════════════════════════════════════════════════
# AKT 专用清洗函数
# ════════════════════════════════════════════════════════════════

# OCR 垃圾字符（src 侧常见）
OCR_GARBAGE_RE = re.compile(r"[°¿¡®©™«»¬¦¤£¥€¢]")

# 学术脚注模式（混入 translation 中）
FOOTNOTE_PATTERNS = [
    # 数字+冒号开头的脚注引用 (如 "11: instead of: ...")
    re.compile(r"\b\d{1,3}:\s+(?:the |instead |still |could |not |end |cf\.|see |or |possibly ).*?(?=[.!?]|$)", re.I),
    # "the reading/restoration ... not certain" 类学术注释
    re.compile(r"(?:the reading|the restoration|could also be read|not certain|possibility would|perhaps read|still visible|not inscribed|cf\. above|see above|see below)[^.!?]*[.!?]?", re.I),
    # 页码/文献引用
    re.compile(r"\bp\.\s*\d+\b"),
    re.compile(r"\bREL\s+\d+\b"),
    re.compile(r"\bpp?\.\s*\d+[-–]\d+\b"),
    # 学术标记
    re.compile(r"\((?:erasure|end of text|not inscribed|erased|broken|rest broken|beginning broken|too broken)[^)]*\)", re.I),
    # "sic" 标记
    re.compile(r"\bsic[_!]?\b", re.I),
    # Upside down / line 标记
    re.compile(r"\(?\bupside down\b\)?", re.I),
    re.compile(r"\b(?:obv|rev|l\.e|u\.e|r\.e|left edge|right edge|upper edge|lower edge)\.\s*", re.I),
    # 行号引用: "I. 18:", "line 5:"
    re.compile(r"\b[IVX]+\.\s*\d+:.*?(?=[.!?]|$)"),
]

# 体裁标签（有时混在翻译开头/结尾）
GENRE_LABEL_RE = re.compile(
    r"(?:^|\s)(?:Blood-money|Legal document|Debt note|Letter|Administrative|"
    r"Sale contract|Purchase|Loan|Court case|Litigation|Judicial|Transport|"
    r"Power of attorney|Caravan|Receipt)[.,]?\s*$",
    re.I | re.MULTILINE
)


def clean_footnotes(text: str) -> str:
    """删除混入翻译中的学术脚注和标记"""
    for pat in FOOTNOTE_PATTERNS:
        text = pat.sub("", text)
    # 删除体裁标签
    text = GENRE_LABEL_RE.sub("", text)
    # 清理残留的多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fix_semiramisunicode_tgt(text: str) -> str:
    """修复 SemiramisUnicode 字体在 translation 中的编码伪影。

    AKT 8 PDF 使用 SemiramisUnicode 字体:
    - Š (U+0160) 被用作开引号（后跟大写字母或空格+大写）
    - š (U+0161) 被用作闭引号（句末、标点前）
    - 但 Akkadian 名字中 Š/š 需要保留（如 AŠšur, Šu-Laban, IŠtar）
    - $ → ṣ, # → ṭ, % → š, ß → š, § → š, !! → šš
    """
    # 0. 先修复 $ # % ß § 在 tgt 中的残留 (SemiramisUnicode font)
    text = text.replace('$', 'ṣ')
    text = text.replace('#', 'ṭ')
    text = text.replace('%', 'š')
    text = text.replace('ß', 'š')
    text = text.replace('§', 'š')
    # !! → šš (常见: A!!ur → Aššur)
    text = text.replace('!!', 'šš')
    # 单独的 ! 在名字中 → š (如 !u-IŠtar → šu-IŠtar)
    text = re.sub(r'!(?=[aeiuáéíúàèìù])', 'š', text)
    text = re.sub(r'(?<=[A-Z])!(?=[A-Z])', 'Š', text)
    # ( 在名字上下文: A(šur → AŠšur
    text = re.sub(r'(?<=[A-Z])\((?=[a-zšṣṭ])', 'Š', text)

    # 1. Š 后直接跟大写英文 (引号): ŠIn → "In, ŠThe → "The
    text = re.sub(r'Š(?=[A-Z][a-z])', '"', text)
    # ŠI → "I (单独的 I)
    text = re.sub(r'Š(?=I\s)', '"', text)

    # 2. š 在句末/引号位置: goldš → gold", ...š → ..."
    text = re.sub(r'š(?=["\'\s]*[.!?,;:)\]])', '"', text)
    # š 在行末
    text = re.sub(r'š\s*$', '"', text)
    # š 后跟空格+大写字母（新句子开始 = 引号结束）
    text = re.sub(r'š(?=\s+[A-Z])', '"', text)

    # 3. 引号规范化: "" → " （清理双引号）
    text = text.replace('""', '"')
    # 弯引号 → 直引号
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    return text


def fix_semiramisunicode_src(text: str) -> str:
    """修复 src 中的 SemiramisUnicode 字体编码伪影。

    AKT PDF 使用 SemiramisUnicode 字体:
    - $ → ṣ/Ṣ
    - # → ṭ/Ṭ
    - % → š (有时是 ē)
    - ! → š/Š (在音节中) 或保持 (作为 collation mark)
    - ß → š (German encoding)
    - § → š (German encoding)
    - ( → 有时被 OCR 误读
    """
    # $ → ṣ (最常见: $í → ṣí, $a → ṣa)
    text = re.sub(r'\$(?=[aeiuAEIUáéíúàèìù])', 'ṣ', text)
    text = re.sub(r'\$(?=[^a-zA-Z]|$)', 'ṣ', text)  # 词尾 $
    text = text.replace('$', 'ṣ')  # 剩余 $

    # # → ṭ
    text = re.sub(r'#(?=[aeiuAEIUáéíúàèìù])', 'ṭ', text)
    text = text.replace('#', 'ṭ')

    # % → š (most common in SemiramisUnicode)
    text = text.replace('%', 'š')

    # ß → š, § → š (German publication encoding)
    text = text.replace('ß', 'š')
    text = text.replace('§', 'š')

    # ! in transliteration context → š (SemiramisUnicode)
    # But ! is also used as a collation mark in Assyriology
    # Strategy: ! between syllable characters → š
    # !a, !e, !i, !u → ša, še, ši, šu
    text = re.sub(r'!(?=[aeiuáéíúàèìù])', 'š', text)
    # A! → AŠ (but careful: could be legitimate !)
    text = re.sub(r'(?<=[A-Z])!(?=[A-Z])', 'Š', text)  # e.g., A!UR → AŠUR

    # " (double quote) used as Š in SemiramisUnicode src
    # Pattern: letter-"syllable (e.g., A-"ur → A-šur, I"tar → Ištar)
    text = re.sub(r'(?<=[-.])"(?=[a-záéíúàèìù])', 'š', text)
    text = re.sub(r'(?<=[-.])"(?=[A-ZÁÉÍÚÀÈÌÙ])', 'Š', text)
    # " after uppercase letter before consonant (e.g., I"tar, A"ur)
    text = re.sub(r'(?<=[A-Z])"(?=[a-z])', 'š', text)

    return text


def src_has_too_much_noise(text: str) -> bool:
    """检查 src 是否含过多 OCR 噪声或非 Akkadian 字符"""
    if OCR_GARBAGE_RE.search(text):
        return True
    # 过多非 Akkadian 字符 (允许: 拉丁字母+变音+连字符+数字+括号+点+空格+<gap>+引号+冒号)
    clean = re.sub(r'[a-zA-ZÀ-öø-ÿšŠṣṢṭṬḫḪ0-9\s\-\.\,\(\)\{\}<>½¼¾⅓⅔⅙⅚⅝₀-₉:;/\'\"!]', '', text)
    # 如果残留非标准字符超过 src 长度的 10%，认为噪声
    if len(clean) > max(3, len(text) * 0.1):
        return True
    return False


# ════════════════════════════════════════════════════════════════
# 主流程: 清洗 akt_extracted.csv
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("清洗 akt_extracted.csv")
print("=" * 60)

df = pd.read_csv(HERE / "akt_extracted.csv", encoding="utf-8")
print(f"原始行数: {len(df)}")
print(f"来源分布:\n{df['source_pdf'].value_counts().to_string()}\n")

stats = {"原始": len(df)}

# Step 1: 去除 NaN/空
df = df.dropna(subset=["transliteration", "translation"])
df = df[df["transliteration"].astype(str).str.strip() != ""]
df = df[df["translation"].astype(str).str.strip() != ""]
stats["去空后"] = len(df)
print(f"去空后: {len(df)}")

# Step 2: 最小词数过滤
df["src_words"] = df["transliteration"].astype(str).str.split().str.len()
df["tgt_words"] = df["translation"].astype(str).str.split().str.len()
df = df[(df["src_words"] >= 4) & (df["tgt_words"] >= 4)]
stats[">=4词后"] = len(df)
print(f">=4词过滤后: {len(df)}")

# Step 2.5: SemiramisUnicode src 编码修复
df["transliteration"] = df["transliteration"].astype(str).apply(fix_semiramisunicode_src)

# Step 2.6: 清理 src 中混入的体裁标签和英文学术注释
_SRC_GENRE_RE = re.compile(
    r"\s*(?:Blood-money|Legal document|Debt note|Letter|Administrative|"
    r"Sale contract|Purchase|Loan|Court case|Litigation|Judicial|Transport|"
    r"Power of attorney|Caravan|Receipt)[.,]?\s*",
    re.I
)
_SRC_ACADEMIC_NOTE_RE = re.compile(
    r"(?:the reading|the restoration|could also|not certain|possibility|perhaps read|"
    r"still visible|not inscribed|cf\. above|see above|instead of|would have been|"
    r"one might|Dercksen suggested|he warns)[^.]*\.?",
    re.I
)
_SRC_PAGE_REF_RE = re.compile(r"\bp\.\s*\d+\b|\bpp?\.\s*\d+[-–]\d+\b")
_SRC_HORIZONTAL_RE = re.compile(r"horizontal[^\s]*\s+", re.I)

def clean_src_annotations(text):
    text = _SRC_GENRE_RE.sub(" ", text)
    text = _SRC_ACADEMIC_NOTE_RE.sub(" ", text)
    text = _SRC_PAGE_REF_RE.sub(" ", text)
    text = _SRC_HORIZONTAL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

df["transliteration"] = df["transliteration"].apply(clean_src_annotations)

# Step 3: 学术脚注清理 + 体裁标签清理
df["translation_clean"] = df["translation"].astype(str).apply(clean_footnotes)
# 重新检查清理后词数
df["tgt_words_clean"] = df["translation_clean"].str.split().str.len()
df = df[df["tgt_words_clean"] >= 4]
stats["脚注清理后"] = len(df)
print(f"脚注清理后: {len(df)}")

# Step 4: SemiramisUnicode 编码修复 (translation)
df["translation_clean"] = df["translation_clean"].apply(fix_semiramisunicode_tgt)

# Step 5: OCR 噪声过滤 (src)
noise_mask = df["transliteration"].astype(str).apply(src_has_too_much_noise)
df = df[~noise_mask]
stats["OCR过滤后"] = len(df)
print(f"OCR噪声过滤后: {len(df)}")

# Step 6: 通用噪声过滤 (src + tgt)
noisy_src = df["transliteration"].astype(str).apply(is_noisy)
noisy_tgt = df["translation_clean"].apply(is_noisy)
df = df[~noisy_src & ~noisy_tgt]
stats["通用噪声后"] = len(df)
print(f"通用噪声过滤后: {len(df)}")

# Step 7: 元注释过滤
meta_mask = df["translation_clean"].apply(is_meta)
df = df[~meta_mask]
stats["元注释后"] = len(df)
print(f"元注释过滤后: {len(df)}")

# Step 8: 应用标准预处理
df["src_processed"] = df["transliteration"].astype(str).apply(preprocess_src)
df["tgt_processed"] = df["translation_clean"].apply(preprocess_tgt)

# Step 9: 预处理后再次检查空/短
df = df[df["src_processed"].str.strip() != ""]
df = df[df["tgt_processed"].str.strip() != ""]
df = df[df["src_processed"].str.split().str.len() >= 4]
df = df[df["tgt_processed"].str.split().str.len() >= 4]
stats["预处理后"] = len(df)
print(f"预处理后: {len(df)}")

# Step 10: 长度比过滤（极端对齐问题）
src_len = df["src_processed"].str.split().str.len()
tgt_len = df["tgt_processed"].str.split().str.len()
ratio = src_len / tgt_len.clip(lower=1)
df = df[(ratio >= 0.15) & (ratio <= 8)]
stats["长度比后"] = len(df)
print(f"长度比过滤后: {len(df)}")

# Step 11: 去重
df = df.drop_duplicates(subset=["src_processed"])
stats["去重后"] = len(df)
print(f"去重后: {len(df)}")

# 输出
out_df = df[["source_pdf", "transliteration", "translation_clean", "src_processed", "tgt_processed"]].copy()
out_df = out_df.rename(columns={"translation_clean": "translation"})
out_df = out_df.reset_index(drop=True)
out_df.to_csv(HERE / "akt_cleaned.csv", index=False, encoding="utf-8")

print(f"\n保存: akt_cleaned.csv ({len(out_df)} 行)")
print(f"\n来源分布:")
print(out_df["source_pdf"].value_counts().to_string())

print(f"\n清洗统计:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# 质量样本
print(f"\n随机样本 (5 条):")
for i, (_, r) in enumerate(out_df.sample(min(5, len(out_df)), random_state=42).iterrows()):
    print(f"  [{i}] src: {r['src_processed'][:100]}")
    print(f"      tgt: {r['tgt_processed'][:150]}")
    print()


# ════════════════════════════════════════════════════════════════
# 修复 akt8_extracted.csv 编码问题
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("修复 akt8_extracted.csv 编码")
print("=" * 60)

akt8 = pd.read_csv(HERE / "akt8_extracted.csv", encoding="utf-8")
print(f"原始行数: {len(akt8)}")

# 编码修复
akt8["translation"] = akt8["translation"].astype(str).apply(fix_semiramisunicode_tgt)

# 应用标准预处理
akt8["src_processed"] = akt8["transliteration"].astype(str).apply(preprocess_src)
akt8["tgt_processed"] = akt8["translation"].apply(preprocess_tgt)

# 过滤
akt8 = akt8.dropna(subset=["src_processed", "tgt_processed"])
akt8 = akt8[akt8["src_processed"].str.strip() != ""]
akt8 = akt8[akt8["tgt_processed"].str.strip() != ""]
akt8 = akt8[akt8["src_processed"].str.split().str.len() >= 4]
akt8 = akt8[akt8["tgt_processed"].str.split().str.len() >= 4]
akt8 = akt8.drop_duplicates(subset=["src_processed"])
akt8 = akt8.reset_index(drop=True)

akt8.to_csv(HERE / "akt8_cleaned.csv", index=False, encoding="utf-8")
print(f"保存: akt8_cleaned.csv ({len(akt8)} 行)")

# 编码修复前后对比样本
print(f"\n编码修复样本:")
for i, (_, r) in enumerate(akt8.head(3).iterrows()):
    print(f"  [{i}] tgt: {r['tgt_processed'][:200]}")
    print()

# ════════════════════════════════════════════════════════════════
# 清洗 akt6ab_extracted.csv (AKT 6a + 6b, 标准字体, 无 SemiramisUnicode)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("清洗 akt6ab_extracted.csv")
print("=" * 60)

akt6_path = HERE / "akt6ab_extracted.csv"
if akt6_path.exists():
    df6 = pd.read_csv(akt6_path, encoding="utf-8")
    print(f"原始行数: {len(df6)}")
    print(f"来源分布:\n{df6['source_pdf'].value_counts().to_string()}\n")

    stats6 = {"原始": len(df6)}

    # Step 1: 去除 NaN/空
    df6 = df6.dropna(subset=["transliteration", "translation"])
    df6 = df6[df6["transliteration"].astype(str).str.strip() != ""]
    df6 = df6[df6["translation"].astype(str).str.strip() != ""]
    stats6["去空后"] = len(df6)

    # Step 2: 最小词数过滤
    df6["src_words"] = df6["transliteration"].astype(str).str.split().str.len()
    df6["tgt_words"] = df6["translation"].astype(str).str.split().str.len()
    df6 = df6[(df6["src_words"] >= 4) & (df6["tgt_words"] >= 4)]
    stats6[">=4词后"] = len(df6)
    print(f">=4词过滤后: {len(df6)}")

    # AKT 6a/6b 字体编码修复 (非 SemiramisUnicode, 但有其他字体伪影)
    def fix_akt6_font_encoding(text: str) -> str:
        """修复 AKT 6a/6b PDF 字体编码伪影"""
        text = str(text)
        # £ (U+00A3) → í (i with acute, 最常见: q£-bi → qí-bi)
        text = text.replace('£', 'í')
        # «» (guillemets) → 编辑注释括号, 保留内容
        text = text.replace('«', '').replace('»', '')
        # · (middle dot) → - (hyphen, 用于音节分隔: il5·qe → il5-qe)
        text = text.replace('·', '-')
        # • (bullet) 的处理
        text = re.sub(r'\[?\s*•{2,}\s*\]?', ' <gap> ', text)  # ••• → gap
        text = text.replace('•', '')  # 单个 bullet 删除
        # 清理 OCR 伪影: 弯引号 → 直引号
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        return text

    df6["transliteration"] = df6["transliteration"].astype(str).apply(fix_akt6_font_encoding)
    df6["translation"] = df6["translation"].astype(str).apply(fix_akt6_font_encoding)

    # 清理体裁标签和学术注释
    df6["transliteration"] = df6["transliteration"].apply(clean_src_annotations)

    # Step 3: 学术脚注清理 + 体裁标签清理 (translation)
    df6["translation_clean"] = df6["translation"].astype(str).apply(clean_footnotes)
    df6["tgt_words_clean"] = df6["translation_clean"].str.split().str.len()
    df6 = df6[df6["tgt_words_clean"] >= 4]
    stats6["脚注清理后"] = len(df6)
    print(f"脚注清理后: {len(df6)}")

    # Step 4: OCR 噪声过滤
    noise_mask6 = df6["transliteration"].astype(str).apply(src_has_too_much_noise)
    df6 = df6[~noise_mask6]
    stats6["OCR过滤后"] = len(df6)
    print(f"OCR噪声过滤后: {len(df6)}")

    # Step 5: 通用噪声过滤
    noisy_src6 = df6["transliteration"].astype(str).apply(is_noisy)
    noisy_tgt6 = df6["translation_clean"].apply(is_noisy)
    df6 = df6[~noisy_src6 & ~noisy_tgt6]
    stats6["通用噪声后"] = len(df6)
    print(f"通用噪声过滤后: {len(df6)}")

    # Step 6: 元注释过滤
    meta_mask6 = df6["translation_clean"].apply(is_meta)
    df6 = df6[~meta_mask6]
    stats6["元注释后"] = len(df6)
    print(f"元注释过滤后: {len(df6)}")

    # Step 7: 标准预处理
    df6["src_processed"] = df6["transliteration"].astype(str).apply(preprocess_src)
    df6["tgt_processed"] = df6["translation_clean"].apply(preprocess_tgt)

    # Step 8: 预处理后检查空/短
    df6 = df6[df6["src_processed"].str.strip() != ""]
    df6 = df6[df6["tgt_processed"].str.strip() != ""]
    df6 = df6[df6["src_processed"].str.split().str.len() >= 4]
    df6 = df6[df6["tgt_processed"].str.split().str.len() >= 4]
    stats6["预处理后"] = len(df6)
    print(f"预处理后: {len(df6)}")

    # Step 9: 长度比过滤
    src_len6 = df6["src_processed"].str.split().str.len()
    tgt_len6 = df6["tgt_processed"].str.split().str.len()
    ratio6 = src_len6 / tgt_len6.clip(lower=1)
    df6 = df6[(ratio6 >= 0.15) & (ratio6 <= 8)]
    stats6["长度比后"] = len(df6)
    print(f"长度比过滤后: {len(df6)}")

    # Step 10: 去重 (自身去重)
    df6 = df6.drop_duplicates(subset=["src_processed"])
    stats6["自身去重后"] = len(df6)

    # Step 11: 跨数据集去重 (与 akt_cleaned 去重)
    akt_existing = pd.read_csv(HERE / "akt_cleaned.csv", encoding="utf-8")
    existing_srcs = set(akt_existing["src_processed"].values)
    before = len(df6)
    df6 = df6[~df6["src_processed"].isin(existing_srcs)]
    stats6["跨数据集去重后"] = len(df6)
    print(f"跨数据集去重后: {len(df6)} (去除 {before - len(df6)} 重复)")

    # 输出
    out_df6 = df6[["source_pdf", "transliteration", "translation_clean", "src_processed", "tgt_processed"]].copy()
    out_df6 = out_df6.rename(columns={"translation_clean": "translation"})
    out_df6 = out_df6.reset_index(drop=True)
    out_df6.to_csv(HERE / "akt6ab_cleaned.csv", index=False, encoding="utf-8")

    print(f"\n保存: akt6ab_cleaned.csv ({len(out_df6)} 行)")
    print(f"\n来源分布:")
    print(out_df6["source_pdf"].value_counts().to_string())

    print(f"\n清洗统计:")
    for k, v in stats6.items():
        print(f"  {k}: {v}")

    # 质量样本
    print(f"\n随机样本 (5 条):")
    for i, (_, r) in enumerate(out_df6.sample(min(5, len(out_df6)), random_state=42).iterrows()):
        print(f"  [{i}] src: {r['src_processed'][:100]}")
        print(f"      tgt: {r['tgt_processed'][:150]}")
        print()
else:
    print(f"跳过: {akt6_path} 不存在 (先运行 extract_akt6.py)")


print("\n完成！")
