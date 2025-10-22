import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# Quickly prints dataset shape, feature types, and missingness summary.
def inspect_dataset(file_name: str):
    df = pd.read_csv(DATA_DIR / file_name)
    print(f"\n=== {file_name} ===")
    print(f"Shape: {df.shape}")
    print("\nColumn Types:")
    print(df.dtypes.value_counts())

    print("\nMissing Values (%):")
    missing = df.isna().sum() / len(df) * 100
    print(missing[missing > 0].round(2).to_string())

    print("\nSample Rows:")
    print(df.head(3))

if __name__ == "__main__":
    for csv in ["iris.csv", "wine_quality.csv", "adult_income.csv"]:
        inspect_dataset(csv)
