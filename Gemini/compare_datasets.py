import pandas as pd

def main():
    with open("dataset_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("=== Analyzing Consistency Between Primary and Extra Datasets ===\n\n")
        
        # Load both datasets
        df_primary = pd.read_csv("train_cleaned_v2.csv")
        df_extra = pd.read_csv("train_extra_cleaned.csv")
        
        f.write(f"Primary Dataset Rows: {len(df_primary)}\n")
        f.write(f"Extra Dataset Rows: {len(df_extra)}\n\n")
        
        # 1. Check text length distributions
        for name, df in [("Primary", df_primary), ("Extra", df_extra)]:
            src_len = df["transliteration"].astype(str).str.split().str.len()
            tgt_len = df["translation"].astype(str).str.split().str.len()
            f.write(f"--- {name} Dataset Lengths (Words) ---\n")
            f.write(f"Source Mean: {src_len.mean():.1f}, Median: {src_len.median():.1f}\n")
            f.write(f"Target Mean: {tgt_len.mean():.1f}, Median: {tgt_len.median():.1f}\n\n")
            
        # 2. Sample comparison
        f.write("--- Primary Dataset Samples ---\n")
        for i in range(3):
            sample = df_primary.sample(1, random_state=42+i).iloc[0]
            f.write(f"SRC: {sample['transliteration']}\n")
            f.write(f"TGT: {sample['translation']}\n\n")
            
        f.write("--- Extra Dataset Samples ---\n")
        for i in range(3):
            sample = df_extra.sample(1, random_state=42+i).iloc[0]
            f.write(f"SRC: {sample['transliteration']}\n")
            f.write(f"TGT: {sample['translation']}\n\n")

if __name__ == "__main__":
    main()
