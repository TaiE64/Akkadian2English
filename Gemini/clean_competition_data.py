import pandas as pd
import re
import os
import argparse
from pathlib import Path

# =========================
# 字符转换及正则模式
# =========================
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})
_CHAR_MAP = {"ḫ":"h","Ḫ":"H","ʾ":"","₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
             "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"}
_CHAR_TRANS = str.maketrans(_CHAR_MAP)

# 破损和间隙模式
_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+|……|\[\.+\])")
_BRACKET_X_RE = re.compile(r"(\[\s*x\s*\]|\(\s*x\s*\))")
_XTOKEN_RUN_RE = re.compile(r"\bx(?:\s+x)+\b", re.I)
_XRUN_RE = re.compile(r"(?<!\w)x{2,}(?!\w)", re.I)
_XTOK_RE = re.compile(r"(?<!\w)x(?!\w)", re.I)
_DET_PARENS_RE = re.compile(r"\(([A-Za-z0-9]{1,4})\)") # 限定词转换: (d) -> {d}

# 分数转换 (针对 Target English)
_FRACTIONS = {
    r'\.5\b': '½', r'\.25\b': '¼', r'\.75\b': '¾',
    r'\.33+\d*\b': '⅓', r'\.66+\d*\b': '⅔',
    r'\.16+\d*\b': '⅙', r'\.83+\d*\b': '⅚',
    r'\.125\b': '⅛', r'\.375\b': '⅜', r'\.625\b': '⅝', r'\.875\b': '⅞'
}

# 噪音清理
_NOISE_RE = re.compile(r"\((?:fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)", re.I)
_WS_RE = re.compile(r"\s+")

def clean_source_akkadian(text):
    """清理阿卡德语音译 (Source)"""
    if pd.isna(text) or not str(text).strip():
        return ""
    
    s = str(text)
    
    # 变音符转换 sz -> š 等
    s = s.replace("sz", "š").replace("SZ", "Š").replace("s,", "ṣ").replace("S,", "Ṣ").replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    
    # 限定词括号: (d) -> {d}
    s = _DET_PARENS_RE.sub(r"{\1}", s)
    
    # 统一所有的破损为 <gap>
    s = _XTOKEN_RUN_RE.sub("<gap>", s)
    s = _ELLIPSIS_RE.sub("<gap>", s)
    s = _BRACKET_X_RE.sub("<gap>", s)
    s = _XRUN_RE.sub("<gap>", s)
    s = _XTOK_RE.sub("<gap>", s)
    
    # 清理遗留的连贯 big_gap
    s = s.replace("<big_gap>", "<gap>")
    s = s.replace("<gap> <gap>", "<gap>")
    s = s.replace("-<gap> <gap>", "-<gap>")
    
    # 去括号保留内容
    s = s.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    # 但恢复 <gap> 标记
    s = s.replace("gap", "<gap>") 
    
    # 字符转换 (去下标, H统一)
    s = s.translate(_CHAR_TRANS).replace("ₓ", "")
    
    s = _WS_RE.sub(" ", s).strip()
    return s

def clean_target_english(text):
    """清理目标英语翻译 (Target)"""
    if pd.isna(text) or not str(text).strip():
        return ""
    
    s = str(text)
    
    # 噪音标签清理
    s = _NOISE_RE.sub("", s)
    
    # 测试集小数被截断，并且只有 Unicode 分数。训练集也必须将小数转为 Unicode
    for pat, rep in _FRACTIONS.items():
        s = re.sub(r'(\d+)' + pat, r'\1' + rep, s)
        s = re.sub(r'\b0' + pat, rep, s) # 0.5 -> ½ (无前导数字)
    
    # 清理遗留噪音符号
    bad_chars = '!?()—–<>⌈⌋⌊[]+/'
    s = s.translate(str.maketrans('', '', bad_chars))
    
    # 规范化多余空格
    s = _WS_RE.sub(" ", s).strip()
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw train.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned csv")
    parser.add_argument("--extra_input", type=str, default=None, help="Path to extra train_clean_v1.csv dataset")
    parser.add_argument("--extra_output", type=str, default=None, help="Path to save cleaned extra dataset")
    args = parser.parse_args()
    
    # 1. Clean primary competition data
    print(f"Loading primary data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Initial shape: {df.shape}")
    
    #... rest of cleaning for df ...
    df["source_cleaned"] = df["transliteration"].apply(clean_source_akkadian)
    df["target_cleaned"] = df["translation"].apply(clean_target_english)
    
    df_clean = df[~df["source_cleaned"].str.strip().eq("") & ~df["target_cleaned"].str.strip().eq("")]
    df_final = df_clean.drop(columns=["transliteration", "translation"]).rename(
        columns={"source_cleaned": "transliteration", "target_cleaned": "translation"}
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_final.to_csv(args.output, index=False)
    print(f"Cleaned primary dataset saved to {args.output} (Rows: {len(df_final)})")

    # 2. Clean extra data if provided
    if args.extra_input and args.extra_output:
        print(f"\nLoading extra data from {args.extra_input}...")
        df_ext = pd.read_csv(args.extra_input)
        print(f"Extra data initial shape: {df_ext.shape}")
        
        # Determine source/target columns. train_clean_v1 might have different names, usually transliteration/translation
        src_col = "transliteration" if "transliteration" in df_ext.columns else "source"
        tgt_col = "translation" if "translation" in df_ext.columns else "target"
        
        # Filter: If it has 'is_oa' or 'text_type', we only want syllabic.
        if "text_type" in df_ext.columns:
            df_ext = df_ext[df_ext["text_type"].str.lower() != "normalized"]
            print(f"After removing 'normalized' text: {df_ext.shape}")
            
        df_ext["source_cleaned"] = df_ext[src_col].apply(clean_source_akkadian)
        df_ext["target_cleaned"] = df_ext[tgt_col].apply(clean_target_english)
        
        df_ext_clean = df_ext[~df_ext["source_cleaned"].str.strip().eq("") & ~df_ext["target_cleaned"].str.strip().eq("")]
        
        df_ext_final = pd.DataFrame({
            "transliteration": df_ext_clean["source_cleaned"],
            "translation": df_ext_clean["target_cleaned"]
        })
        
        os.makedirs(os.path.dirname(args.extra_output), exist_ok=True)
        df_ext_final.to_csv(args.extra_output, index=False)
        print(f"Cleaned extra dataset saved to {args.extra_output} (Rows: {len(df_ext_final)})")

if __name__ == "__main__":
    main()
