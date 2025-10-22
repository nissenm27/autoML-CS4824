import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocessing.schema import infer_schema
from preprocessing.build_preprocessor import build_preprocessor
from models.linear_models import LogisticRegressionModel
from search.grid_random import SearchManager
from search.search_utils import get_param_grid

if __name__ == "__main__":
    # Dataset
    df = pd.read_csv("data/adult_income.csv")
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]

    # Build preprocessing
    schema = infer_schema(X)
    pre = build_preprocessor(schema)

    # Choose model & param grid
    model = LogisticRegressionModel()
    param_grid = get_param_grid("logistic_regression")

    # Create pipeline
    pipe = Pipeline([
        ("prep", pre),
        ("model", model.model),
    ])

    # Run Grid Search
    searcher = SearchManager(search_type="grid", param_grid=param_grid, scoring="accuracy", cv=3)
    best_model, best_params, best_score = searcher.run(pipe, X, y)

    print("\nBest Params:", best_params)
    print("Best CV Score:", round(best_score, 4))