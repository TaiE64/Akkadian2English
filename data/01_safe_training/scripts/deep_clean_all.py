"""Comprehensive deep clean of merged_all.csv."""
import pandas as pd
import re
import unicodedata

path = r'C:\Users\29421\Desktop\kaggle_Challenge\data\01_safe_training\merged_all.csv'
df = pd.read_csv(path)
start = len(df)
print(f'Starting: {start} rows')
removed_reasons = {}

# === 1. Remove NaN / empty rows ===
mask = df['transliteration'].isna() | df['translation'].isna() | \
       (df['transliteration'].astype(str).str.strip() == '') | \
       (df['translation'].astype(str).str.strip() == '')
n = mask.sum()
if n: removed_reasons['Empty/NaN'] = n
df = df[~mask].copy()
print(f'1. Empty/NaN: removed {n}, remaining {len(df)}')

# === 2. Strip whitespace ===
df['transliteration'] = df['transliteration'].astype(str).str.strip()
df['translation'] = df['translation'].astype(str).str.strip()

# === 3. Remove rows with non-Latin characters ===
# Block: control chars, Ethiopic, Syriac, Arabic, Hebrew, Greek, Cyrillic, CJK, etc.
BAD_CHAR_RE = re.compile(
    '['
    '\u0080-\u009F'     # C1 control chars
    '\u00A7'            # §
    '\u0370-\u03FF'     # Greek
    '\u0400-\u04FF'     # Cyrillic
    '\u0500-\u052F'     # Cyrillic Supplement
    '\u0590-\u05FF'     # Hebrew
    '\u0600-\u06FF'     # Arabic
    '\u0700-\u074F'     # Syriac
    '\u0900-\u097F'     # Devanagari
    '\u1200-\u137F'     # Ethiopic
    '\u2000-\u200F'     # General punctuation (zero-width chars etc)
    '\u2028-\u202F'     # Line/paragraph separators
    '\u2060-\u206F'     # Invisible formatters
    '\u4E00-\u9FFF'     # CJK
    '\uFB50-\uFDFF'     # Arabic Presentation Forms
    '\uFE70-\uFEFF'     # Arabic Presentation Forms B
    '\uFEFF'            # BOM
    '\uFFF0-\uFFFF'     # Specials
    ']'
)
mask = df['transliteration'].apply(lambda x: bool(BAD_CHAR_RE.search(str(x)))) | \
       df['translation'].apply(lambda x: bool(BAD_CHAR_RE.search(str(x))))
n = mask.sum()
if n: removed_reasons['Non-Latin chars'] = n
df = df[~mask].copy()
print(f'2. Non-Latin chars: removed {n}, remaining {len(df)}')

# === 4. Remove ratio anomalies ===
df['_src_len'] = df['transliteration'].str.len()
df['_tgt_len'] = df['translation'].str.len()
df['_ratio'] = df['_tgt_len'] / df['_src_len'].clip(lower=1)
lo, hi = 0.15, 5.0
mask = ~((df['_ratio'] >= lo) & (df['_ratio'] <= hi))
n = mask.sum()
if n: removed_reasons[f'Ratio outside [{lo},{hi}]'] = n
df = df[~mask].copy()
print(f'3. Ratio anomalies: removed {n}, remaining {len(df)}')

# === 5. Remove very short rows (< 5 chars in transliteration) ===
mask = df['_src_len'] < 5
n = mask.sum()
if n: removed_reasons['Too short (<5 chars)'] = n
df = df[~mask].copy()
print(f'4. Too short: removed {n}, remaining {len(df)}')

# === 6. Clean up text (character-level fixes, not row removal) ===
# Remove stray ? from transliteration
df['transliteration'] = df['transliteration'].str.replace('?', '', regex=False)
# Normalize multiple spaces
df['transliteration'] = df['transliteration'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['translation'] = df['translation'].str.replace(r'\s+', ' ', regex=True).str.strip()
# Remove [Ø]
df['transliteration'] = df['transliteration'].str.replace('[Ø]', '', regex=False)
df['translation'] = df['translation'].str.replace('[Ø]', '', regex=False)
print(f'5. Text cleanup done')

# === 7. Deduplicate on transliteration ===
before_dedup = len(df)
df = df.drop_duplicates(subset=['transliteration'], keep='first')
n = before_dedup - len(df)
if n: removed_reasons['Duplicates'] = n
print(f'6. Dedup: removed {n}, remaining {len(df)}')

# === 8. Remove rows where transliteration == translation (copy errors) ===
mask = df['transliteration'] == df['translation']
n = mask.sum()
if n: removed_reasons['src == tgt'] = n
df = df[~mask].copy()
print(f'7. src==tgt: removed {n}, remaining {len(df)}')

# Drop helper columns and save
df = df[['transliteration', 'translation']]
df.to_csv(path, index=False)

print(f'\n{"="*50}')
print(f'SUMMARY: {start} -> {len(df)} (removed {start - len(df)} total)')
print(f'{"="*50}')
for reason, count in sorted(removed_reasons.items(), key=lambda x: -x[1]):
    print(f'  {reason}: {count}')
print(f'\nSaved!')
