"""Fast scan for Greek/German/unusual chars using vectorized regex."""
import pandas as pd
import re

path = r'C:\Users\29421\Desktop\kaggle_Challenge\data\01_safe_training\merged_all.csv'
df = pd.read_csv(path)
print(f'Rows: {len(df)}')

# Greek: U+0370-03FF, German umlauts are normal Latin (äöü already in Latin-1)
# Check for Greek specifically
greek = re.compile('[\u0370-\u03FF]')
g_src = df['transliteration'].str.contains(greek, na=False).sum()
g_tgt = df['translation'].str.contains(greek, na=False).sum()
print(f'Greek: {g_src} in src, {g_tgt} in tgt')

# German-specific chars (ß)
ss = df['transliteration'].str.contains('ß', na=False, regex=False).sum() + \
     df['translation'].str.contains('ß', na=False, regex=False).sum()
print(f'ß (Eszett): {ss}')

# Broad scan: anything outside Latin + diacritics
broad = re.compile('[\u0080-\u009F\u0370-\u03FF\u0400-\u052F\u0590-\u074F\u0900-\u137F\u4E00-\u9FFF\uFB50-\uFFFF\u00A7]')
b_src = df['transliteration'].str.contains(broad, na=False).sum()
b_tgt = df['translation'].str.contains(broad, na=False).sum()
print(f'Any non-Latin: {b_src} in src, {b_tgt} in tgt')

if b_src + b_tgt > 0:
    # Show samples
    mask = df['transliteration'].str.contains(broad, na=False) | df['translation'].str.contains(broad, na=False)
    samples = df[mask].head(5)
    for _, r in samples.iterrows():
        src = str(r['transliteration'])[:60]
        tgt = str(r['translation'])[:60]
        print(f'  src: {src}')
        print(f'  tgt: {tgt}')
        print()
