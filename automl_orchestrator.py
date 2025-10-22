"""
AutoML Orchestrator
Integrates preprocessing, model wrappers, and search algorithms into one automated pipeline.
Produces a leaderboard of model performance across your Model Zoo.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from pathlib import Path
import json

# Import core modules
from preprocessing.schema import infer_schema
from preprocessing.build_preprocessor import build_preprocessor
from models.linear_models import LogisticRegressionModel, RidgeRegressionModel
from models.tree_models import DecisionTreeModel, RandomForestModel, GradientBoostingModel
from models.neural_net import NeuralNetworkModel
from search.grid_random import SearchManager
from search.search_utils import get_param_grid

 
# CONFIG
 

DATA_PATH = "data/adult_income.csv"      # You can swap datasets easily here
TARGET_COLUMN = "class"           # Adjust if dataset differs
SEARCH_TYPE = "grid"              # or "random"
CV_FOLDS = 3
SCORING = "accuracy"

 
# MODEL REGISTRY
 

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel(),
    "ridge": RidgeRegressionModel(),
    "decision_tree": DecisionTreeModel(),
    "random_forest": RandomForestModel(),
    "gradient_boosting": GradientBoostingModel(),
    "neural_net": NeuralNetworkModel(),
}


# TASK DETECTOR


def detect_task_type(y):
    """
    Lightweight task detector.
    Returns 'classification' if target is categorical or has few unique values,
    otherwise 'regression'.
    """
    # object dtype or small number of unique discrete values → classification
    if y.dtype == "object" or len(y.unique()) < 20:
        return "classification"
    else:
        return "regression"

 
# MAIN ORCHESTRATION
 

def run_automl(data_path=DATA_PATH, target_column=TARGET_COLUMN):
    print("\n Starting AutoML Pipeline\n")

    # Load dataset 
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    task_type = detect_task_type(y)
    print(f"\n Detected Task Type: {task_type.upper()}")

    # Train/test split (for final validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Build preprocessing pipeline
    schema = infer_schema(X_train)
    preprocessor = build_preprocessor(schema)

    leaderboard = []
    
    # Path for saving results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    start_time_total = time.time()

    for model_name, model_obj in MODEL_REGISTRY.items():
        # Skip incompatible models
        if model_obj.task_type != task_type:
            print(f"\n Skipping {model_name.upper()} ({model_obj.task_type} model not compatible with {task_type} task)")
            continue

        print(f"\n=== Running {model_name.upper()} ===")

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model_obj.model),
        ])

        param_grid = get_param_grid(model_name)
        searcher = SearchManager(
            search_type=SEARCH_TYPE,
            param_grid=param_grid,
            scoring=SCORING,
            cv=CV_FOLDS,
        )

        # --- Runtime tracking for each model ---
        start_time = time.time()
        best_estimator, best_params, best_score = searcher.run(pipe, X_train, y_train)
        runtime = round(time.time() - start_time, 2)

        # Evaluate on held-out test set
        test_score = best_estimator.score(X_test, y_test)

        leaderboard.append({
            "dataset": Path(data_path).stem,
            "task_type": task_type,
            "model": model_name,
            "best_cv_score": round(best_score, 4),
            "test_score": round(test_score, 4),
            "runtime_sec": runtime,
            "best_params": best_params,
        })

    # --- Save leaderboard ---
    total_runtime = round(time.time() - start_time_total, 2)
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="test_score", ascending=False)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"leaderboard_{Path(data_path).stem}_{timestamp}.csv"
    leaderboard_df.to_csv(csv_path, index=False)

    print(f"\n AutoML Leaderboard saved to: {csv_path}")
    print(f"Total runtime: {total_runtime}s")
    print(leaderboard_df.to_string(index=False))
     
    # Display Leaderboard
     
    print("\n AutoML Leaderboard\n")
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="test_score", ascending=False)
    print(leaderboard_df.to_string(index=False))

    return leaderboard_df


if __name__ == "__main__":
    leaderboard = run_automl()
