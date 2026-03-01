import pandas as pd

def main():
    print("Combining primary and extra datasets...")
    df_primary = pd.read_csv("train_cleaned_v2.csv")
    df_extra = pd.read_csv("train_extra_cleaned.csv")
    
    # Concatenate and shuffle
    df_combined = pd.concat([df_primary, df_extra], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = "train_combined_cleaned.csv"
    df_combined.to_csv(output_path, index=False)
    
    print(f"Combined dataset saved to {output_path} with {len(df_combined)} rows.")

if __name__ == "__main__":
    main()
