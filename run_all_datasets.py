"""
Runs the AutoML orchestrator sequentially across multiple datasets.
Outputs separate leaderboard CSVs for each dataset into /results.
"""

from automl_orchestrator import run_automl

# (dataset_path, target_column)
DATASETS = [
    ("data/iris.csv", "class"), # Classification
    ("data/wine_quality.csv", "quality"), # Regression
    ("data/adult_income.csv", "class") # Classification
]

if __name__ == "__main__":
    for data_path, target_col in DATASETS:
        print("\n" + "=" * 60)
        print(f"Running AutoML on: {data_path}")
        print("=" * 60)
        run_automl(data_path=data_path, target_column=target_col)