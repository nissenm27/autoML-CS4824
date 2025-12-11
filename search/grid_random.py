"""
Implements baseline grid and random search for AutoML.
Supports any sklearn-compatible estimator or wrapped model.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, Literal, Any
import numpy as np
from search.bayesian import BayesianOptimizer

SearchType = Literal["grid", "random", "bayes"]

class SearchManager:
    def __init__(
        self,
        search_type: SearchType = "grid",
        param_grid: Dict[str, Any] | None = None,
        n_iter: int = 20,
        scoring: str = "accuracy",
        cv: int = 3,
        random_state: int = 42,
    ):
        self.search_type = search_type
        self.param_grid = param_grid or {}
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.search = None

        # NEW: track trial history for plotting search trajectories
        self.trial_history = []

    def run(self, pipeline: Pipeline, X, y):
        """Run grid, random, or Bayesian search on given pipeline."""

        # ---------------------------------------------------------
        # GRID SEARCH
        # ---------------------------------------------------------
        if self.search_type == "grid":
            self.search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
                return_train_score=False,
            )
            self.search.fit(X, y)

            # LOG EACH TRIAL
            for i, score in enumerate(self.search.cv_results_["mean_test_score"]):
                params = self.search.cv_results_["params"][i]
                self.trial_history.append({
                    "iteration": i + 1,
                    "score": float(score),
                    "params": params
                })

            return (
                self.search.best_estimator_,
                self.search.best_params_,
                self.search.best_score_,
            )

        # ---------------------------------------------------------
        # RANDOM SEARCH
        # ---------------------------------------------------------
        elif self.search_type == "random":
            self.search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1,
                return_train_score=False,
            )
            self.search.fit(X, y)

            # LOG EACH TRIAL
            for i, score in enumerate(self.search.cv_results_["mean_test_score"]):
                params = self.search.cv_results_["params"][i]
                self.trial_history.append({
                    "iteration": i + 1,
                    "score": float(score),
                    "params": params
                })

            return (
                self.search.best_estimator_,
                self.search.best_params_,
                self.search.best_score_,
            )

        # ---------------------------------------------------------
        # BAYESIAN OPTIMIZATION (Optuna)
        # ---------------------------------------------------------
        elif self.search_type == "bayes":
            bayes = BayesianOptimizer(
                model_name="pipeline",
                model_obj=pipeline,
                param_space=self.param_grid,
                scoring=self.scoring,
                cv=self.cv
            )

            best_estimator, best_params, best_score, bayes_history = bayes.run(
                pipeline, X, y
            )

            # record trials from optuna
            self.trial_history = bayes_history

            return best_estimator, best_params, best_score

        else:
            raise ValueError("search_type must be 'grid', 'random', or 'bayes'")
