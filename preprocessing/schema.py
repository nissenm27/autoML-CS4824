import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Schema:
    numeric: List[str]
    categorical: List[str]
    missing_summary: Dict[str, float]
    n_rows: int
    n_cols: int

def infer_schema(df: pd.DataFrame) -> Schema:
    # Automatically identify numeric and categorical features.
    n_rows, n_cols = df.shape
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in df.columns if col not in numeric]

    missing_summary = (df.isna().sum() / len(df)).to_dict()
    return Schema(
        numeric=numeric,
        categorical=categorical,
        missing_summary=missing_summary,
        n_rows=n_rows,
        n_cols=n_cols
    )

# quick test
if __name__ == "__main__":
    sample = pd.read_csv("data/adult_income.csv")
    schema = infer_schema(sample)
    print(schema)
