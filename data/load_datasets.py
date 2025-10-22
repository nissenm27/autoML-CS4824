import openml
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


# Pull small datasets for benchmarking
def fetch_and_save_datasets():
    datasets = {
        "iris": 61,
        "wine_quality": 287,
        "adult_income": 1590,
    }

    # iterate through each dataset and pull
    for name, did in datasets.items():
        dataset = openml.datasets.get_dataset(did, download_all_files=True)
        df, *_ = dataset.get_data(dataset_format="dataframe")
        csv_path = DATA_DIR / f"{name}.csv" # Storing as csv
        df.to_csv(csv_path, index=False)
        print(f"Saved {name} to {csv_path}")

if __name__ == "__main__":
    fetch_and_save_datasets()
