import pandas as pd

files = [
    "data/manwithacat_enriched/train_enriched.csv",
    "data/manwithacat_combined/combined_akkadian_v2_oracc.csv",
    "data/manwithacat_oracc/train.csv",
    "data/manwithacat_augmented/train_augmented_normalized.csv"
]

print("=== Dataset Schemas ===\n")
for f in files:
    try:
        df = pd.read_csv(f, nrows=2)
        print(f"File: {f}")
        print(f"Columns: {list(df.columns)}")
        if 'text_type' in df.columns:
            print(f"First text_type: {df['text_type'].iloc[0]}")
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {f}: {e}")
