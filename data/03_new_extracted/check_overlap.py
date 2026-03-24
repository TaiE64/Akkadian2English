import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pub_clean = pd.read_csv(os.path.join(ROOT, '03_new_extracted/processed/published_texts_cleaned.csv'))
train = pd.read_csv(os.path.join(ROOT, '01_safe_training/train.csv'))
pub_orig = pd.read_csv(os.path.join(ROOT, 'competition/published_texts.csv'))
at = pd.read_parquet(os.path.join(ROOT, '01_safe_training/akkadian-translation/data/train-00000-of-00001.parquet'))
phuc_oa = pd.read_parquet(os.path.join(ROOT, '01_safe_training/phucthaiv02_oa_sentences/train.parquet'))
phuc_pdf = pd.read_parquet(os.path.join(ROOT, '02_risky_training/phucthaiv02_pdf_extracted/cleaned_phuc_pdf_v2.parquet'))

# Also load alignment_1 if exists
al1_path = os.path.join(ROOT, '06_phucthai_hf/akkadian_english_sentences_alignment/data/train-00000-of-00001.parquet')
if os.path.exists(al1_path):
    al1 = pd.read_parquet(al1_path)
else:
    al1 = None

print(f'published_texts_cleaned: {len(pub_clean)} rows')
print(f'Columns: {pub_clean.columns.tolist()}')
print(f'Has translation: {"translation" in pub_clean.columns}')
print(f'Sample:\n{pub_clean.head(3).to_string()}')

# oare_id overlap
pub_ids = set(pub_clean['oare_id'])
train_ids = set(train['oare_id'])
pub_orig_ids = set(pub_orig['oare_id'])

print(f'\n=== oare_id overlap ===')
print(f'pub_cleaned ids: {len(pub_ids)}')
print(f'  vs train.csv: {len(pub_ids & train_ids)} overlap')
print(f'  vs published_texts.csv: {len(pub_ids & pub_orig_ids)} overlap')

# Is it just published_texts.csv cleaned?
print(f'\npub_cleaned == pub_orig ids? {pub_ids == pub_orig_ids}')
print(f'  pub_cleaned - pub_orig: {len(pub_ids - pub_orig_ids)}')
print(f'  pub_orig - pub_cleaned: {len(pub_orig_ids - pub_ids)}')

# Transliteration text overlap
def norm(s):
    return str(s).strip().lower()

pub_src = set(pub_clean['clean_transliteration'].apply(norm))

# Collect all training src texts
all_train_src = set()
all_train_src |= set(train['transliteration'].apply(norm))
all_train_src |= set(at['transliteration'].apply(norm))
all_train_src |= set(phuc_oa['transliteration'].apply(norm))
all_train_src |= set(phuc_pdf['src'].apply(norm))
if al1 is not None:
    all_train_src |= set(al1['transliteration'].apply(norm))

# Also check prepared_data (the actual training data being used)
prepared_path = os.path.join(ROOT, '../qlora/prepared_data/train.csv')
if os.path.exists(prepared_path):
    prepared = pd.read_csv(prepared_path, escapechar='\\')
    prepared_src = set(prepared['src_processed'].apply(norm))
else:
    prepared_src = set()
    print('WARNING: prepared_data/train.csv not found')

# Also check published_texts_sentences (extracted sentence pairs)
pts_path = os.path.join(ROOT, 'competition/published_texts_sentences.csv')
if os.path.exists(pts_path):
    pts = pd.read_csv(pts_path)
    pts_src = set(pts['src_orig'].apply(norm))
    print(f'\npublished_texts_sentences.csv: {len(pts)} rows')
else:
    pts_src = set()

print(f'\n=== transliteration text overlap ===')
print(f'pub_cleaned unique transliterations: {len(pub_src)}')
print(f'  vs all raw training src: {len(pub_src & all_train_src)}')
print(f'  vs prepared_data/train.csv src: {len(pub_src & prepared_src)}')
if pts_src:
    print(f'  vs published_texts_sentences src: {len(pub_src & pts_src)}')
print(f'  NOT in any training data: {len(pub_src - all_train_src - prepared_src - pts_src)}')

# Check by oare_id: which are in akkadian-translation?
if 'text_id' in at.columns:
    at_ids = set(at['text_id'])
    print(f'\n=== oare_id vs akkadian-translation text_id ===')
    print(f'  overlap: {len(pub_ids & at_ids)}')

# Check by oare_id: which are in phuc_pdf?
if 'text_id' in phuc_pdf.columns:
    pdf_ids = set(phuc_pdf['text_id'])
    print(f'  vs phuc_pdf text_id: {len(pub_ids & pdf_ids)}')

print(f'\n=== CONCLUSION ===')
print(f'This file has ONLY transliterations (no translations).')
print(f'It appears to be a cleaned version of published_texts.csv.')
print(f'{len(pub_ids & train_ids)} of its texts overlap with train.csv by oare_id.')
print(f'{len(pub_ids - train_ids)} texts NOT in train.csv.')
print(f'Without translations, this cannot be used as training data directly.')
