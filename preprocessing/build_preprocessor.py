"""
Preprocessing module skeleton.

This file defines the interface for building a ColumnTransformer-based preprocessor
that can handle numeric and categorical features using the inferred schema.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from preprocessing.schema import Schema

def build_preprocessor(schema: Schema) -> ColumnTransformer:
    """
    Given a Schema object, return a scikit-learn ColumnTransformer that:
      - imputes and scales numeric features
      - imputes and encodes categorical features
    """
    # placeholders for Week 2 implementation
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, schema.numeric),
            ("cat", categorical_pipeline, schema.categorical)
        ],
        remainder="drop"
    )

    return preprocessor
