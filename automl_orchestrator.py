"""
AutoML Orchestrator
Integrates preprocessing, model wrappers, and search algorithms into one automated pipeline.
Produces a leaderboard of model performance across your Model Zoo.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from pathlib import Path
import numpy as np
import json

# Import core modules
from preprocessing.schema import infer_schema
from preprocessing.build_preprocessor import build_preprocessor
from models.linear_models import LogisticRegressionModel, RidgeRegressionModel
from models.tree_models import DecisionTreeModel, RandomForestModel, GradientBoostingModel
from models.neural_net import NeuralNetworkModel
from search.grid_random import SearchManager
from search.search_utils import get_param_grid

# Import ensembles
from ensembles.soft_voting import SoftVotingEnsemble
from ensembles.stacking import StackingEnsemble
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


# Import meta-learning
from meta.meta_features import compute_meta_features
from meta.meta_knn import find_similar_datasets

# Import NAS
from nas.tiny_nas import TinyNAS


# CONFIG
 

DATA_PATH = "data/adult_income.csv" # You can swap datasets easily here
TARGET_COLUMN = "class" # Adjust if dataset differs
SEARCH_TYPE = "bayes" # "grid" or "random" or "bayes"
CV_FOLDS = 3

 
# MODEL REGISTRY
 

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel(),
    "ridge": RidgeRegressionModel(),
    "decision_tree": DecisionTreeModel(),
    "random_forest": RandomForestModel(),
    "gradient_boosting": GradientBoostingModel(),
    "neural_net": NeuralNetworkModel(),
    "tiny_nas": TinyNAS(),
}


# Helper to convert stringified tuples back to real tuples
def try_parse_tuple(x):
    """
    Convert JSON-loaded string like '(32, 16)' back into an actual tuple.
    """
    if isinstance(x, str) and x.startswith("(") and x.endswith(")"):
        try:
            return eval(x)
        except:
            return x
    return x


# Helper for Robust Warm-Start Merge

def merge_warm_start_values(list_of_dicts):
    """
    Merge hyperparameters from similar datasets.
    - Averaging for numeric params
    - Most frequent value for categorical params
    - Skip None or missing fields
    """
    merged = {}

    for d in list_of_dicts:
        for key, val in d.items():
            merged.setdefault(key, []).append(val)

    final = {}

    for key, vals in merged.items():
        # Filter out None
        vals = [try_parse_tuple(v) for v in vals if v is not None]

        # Numeric → average
        if all(isinstance(v, (int, float)) for v in vals):
            final[key] = sum(vals) / len(vals)

        # Tuple (e.g., hidden_layer_sizes) → pick the most common
        elif all(isinstance(v, tuple) for v in vals):
            final[key] = max(set(vals), key=vals.count)

        # Strings → pick most common
        elif all(isinstance(v, str) for v in vals):
            final[key] = max(set(vals), key=vals.count)

        # Mixed types → skip (cannot merge safely)
        else:
            continue

    return final


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

    dataset_name = Path(data_path).stem
    
    task_type = detect_task_type(y)
    # Set dynamic scoring based on detected task
    if task_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "r2"

    print(f"\n Detected Task Type: {task_type.upper()}")
    print(f"\n Scoring metric: {scoring.upper()}")

    # ------------------------------------------
    # META-LEARNING: Compute meta-features
    # ------------------------------------------
    current_meta = compute_meta_features(X, y)

    # Load meta dataset (past hyperparameter performance)
    meta_path = Path("meta/meta_dataset.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta_db = json.load(f)
    else:
        meta_db = {}


    # Attach meta-features into meta_db if missing
    meta_db.setdefault(dataset_name, {})

    # Save meta-features
    meta_db[dataset_name]["meta_features"] = current_meta

    # Persist immediately so find_similar_datasets() sees it
    with open(meta_path, "w") as f:
        json.dump(meta_db, f, indent=2)

    # Find similar datasets
    similar_datasets = find_similar_datasets(current_meta, meta_db, k=3)

    ## Build warm-start dictionary for each model
    warm_start_params = {}

    for model_name in MODEL_REGISTRY.keys():
        warm_values = []

        for ds in similar_datasets:
            params = meta_db.get(ds, {}).get(model_name)

            # ensure it's a dictionary of hyperparameters, not meta_features
            if params and isinstance(params, dict):
                warm_values.append(params)

        # Assign AFTER loop
        warm_start_params[model_name] = (
            merge_warm_start_values(warm_values) if warm_values else None
        )


    print("\n[Meta-Learning] Warm-start hyperparameters:")
    print(json.dumps(warm_start_params, indent=2))

    # Train/test split (for final validation)
    if task_type == "classification":
        strat = y
    else:
        strat = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)


    # Build preprocessing pipeline
    schema = infer_schema(X_train)
    preprocessor = build_preprocessor(schema)

    leaderboard = []
    
    best_model_overall = None
    best_score_overall = -np.inf
    best_params_overall = None


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

        # --------------------------------------------
        # SPECIAL CASE: NAS DOES NOT USE PIPELINES
        # --------------------------------------------
        if model_name == "tiny_nas":
            print("Running TinyNAS architecture search …")

            # ---- Apply preprocessing (transform only) ----
            X_train_pre = preprocessor.fit_transform(X_train)   # fit on train only
            X_test_pre  = preprocessor.transform(X_test)

            # Run NAS on numeric features
            best_estimator, best_params, best_score = model_obj.search(X_train_pre, y_train)

            # Evaluate model on transformed test set
            test_score = best_estimator.score(X_test_pre, y_test)

            # Track the best model for this dataset
            if test_score > best_score_overall:
                best_score_overall = test_score
                best_model_overall = best_estimator
                best_params_overall = best_params

            runtime = None

        else:
            # --------------------------------------------
            # Normal models (grid/random/bayes search)
            # --------------------------------------------

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model_obj.model),
            ])

            param_grid = get_param_grid(model_name, search_type=SEARCH_TYPE, warm_start=warm_start_params.get(model_name) or None)
            searcher = SearchManager(
                search_type=SEARCH_TYPE,
                param_grid=param_grid, 
                scoring=scoring,
                cv=CV_FOLDS,
            )

            # --- Runtime tracking for each model ---
            start_time = time.time()
            best_estimator, best_params, best_score = searcher.run(pipe, X_train, y_train)
            runtime = round(time.time() - start_time, 2)

        # Evaluate on held-out test set
        if model_name == "tiny_nas":
            # already computed test_score above
            pass
        else:
            test_score = best_estimator.score(X_test, y_test)


        leaderboard.append({
            "dataset": Path(data_path).stem,
            "task_type": task_type,
            "model": model_name,
            "best_cv_score": round(best_score, 4),
            "test_score": round(test_score, 4),
            "runtime_sec": runtime,
            "best_params": best_params,
            "best_estimator": best_estimator
        })

        # ------------------------------------------
        # Save search trajectory (grid / random / bayes only)
        # ------------------------------------------
        if model_name != "tiny_nas":     # NAS has no SearchManager
            history_path = results_dir / f"{model_name}_{SEARCH_TYPE}_history.json"
            with open(history_path, "w") as f:
                json.dump(searcher.trial_history, f, indent=2)

        meta_db.setdefault(dataset_name, {})
        meta_db[dataset_name][model_name] = best_params
    # ------------------------------------------
    # Save updated meta-learning database
    # ------------------------------------------
    with open(meta_path, "w") as f:
        json.dump(meta_db, f, indent=2)

    # ----------------------------------------------------
    # Save ONE checkpoint: the best model for this dataset
    # ----------------------------------------------------
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"best_model_{dataset_name}.pkl"

    try:
        import joblib
        joblib.dump(best_model_overall, checkpoint_path)
        print(f"[BEST CHECKPOINT] Saved best model for {dataset_name}: {checkpoint_path}")
    except Exception as e:
        print(f"[BEST CHECKPOINT] Failed to save best model: {e}")

    # ------------------------------------------
    # ENSEMBLE CONSTRUCTION (inside run_automl)
    # ------------------------------------------

    # Sort by CV score
    leaders_sorted = sorted(leaderboard, key=lambda x: x["best_cv_score"], reverse=True)

    # Select top 3 models
    top_k = 3
    top_estimators = [entry["best_estimator"] for entry in leaders_sorted[:top_k]]

    # ---- Soft Voting Ensemble ----
    soft_ensemble = SoftVotingEnsemble(top_estimators)
    soft_pred = soft_ensemble.predict(X_test)

    # Convert numeric predictions → original string labels
    if task_type == "classification":
        classes = top_estimators[0].classes_
        # Only map predictions if model outputs ints (not needed for NAS)
        if isinstance(soft_pred[0], (int, np.integer)):
            soft_pred = classes[soft_pred]


    if task_type == "classification":
        soft_score = accuracy_score(y_test, soft_pred)
    else:
        soft_score = r2_score(y_test, soft_pred)

    leaderboard.append({
        "dataset": Path(data_path).stem,
        "task_type": task_type,
        "model": "soft_voting_ensemble",
        "best_cv_score": None,
        "test_score": round(soft_score, 4),
        "runtime_sec": None,
        "best_params": {"models_used": [m["model"] for m in leaders_sorted[:top_k]]}
    })

    # ---- Stacking Ensemble ----
    stack_ensemble = StackingEnsemble(top_estimators)
    stack_ensemble.fit(X_train, y_train)
    stack_pred = stack_ensemble.predict(X_test)

    # Stacking already returns correct string labels → DO NOT map
    # Ensure dtype compatibility (numpy array)
    stack_pred = np.array(stack_pred)

    if task_type == "classification":
        stack_score = accuracy_score(y_test, stack_pred)
    else:
        stack_score = r2_score(y_test, stack_pred)


    leaderboard.append({
        "dataset": Path(data_path).stem,
        "task_type": task_type,
        "model": "stacking_ensemble",
        "best_cv_score": None,
        "test_score": round(stack_score, 4),
        "runtime_sec": None,
        "best_params": {"models_used": [m["model"] for m in leaders_sorted[:top_k]]}
    })

    # --- Save leaderboard ---
    total_runtime = round(time.time() - start_time_total, 2)
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="test_score", ascending=False)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"leaderboard_{Path(data_path).stem}_{timestamp}.csv"
    # leaderboard_df.to_csv(csv_path, index=False)

    print(f"\n AutoML Leaderboard saved to: {csv_path}")
    print(f"Total runtime: {total_runtime}s")
    print(leaderboard_df.to_string(index=False))
    # Save a standardized summary file for visualization
    # leaderboard_df.to_csv("results/model_zoo_summary.csv", index=False)


    return leaderboard_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lightweight AutoML system.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset.")
    parser.add_argument("--target", type=str, default=None, help="Target column name.")
    parser.add_argument("--all", action="store_true",
                        help="Run AutoML on all default baby datasets.")

    args = parser.parse_args()

    if args.all:
        baby_datasets = [
            ("data/adult_income.csv", "class"),
            ("data/iris.csv", "class"),
            ("data/wine_quality.csv", "quality"),
        ]

        all_results = []
        for path, target in baby_datasets:
            print("\n\n==============================")
            print(f"Running AutoML on dataset: {path}")
            print("==============================")
            df = run_automl(data_path=path, target_column=target)
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("results/model_zoo_summary.csv", index=False)
        print("\n=== Combined Model Zoo Summary Saved ===")
        print(combined)

    else:
        # Single dataset mode
        if args.data is None or args.target is None:
            raise ValueError(
                "Must supply --data and --target OR use --all flag."
            )
        run_automl(data_path=args.data, target_column=args.target)

