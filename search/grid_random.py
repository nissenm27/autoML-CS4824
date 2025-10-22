"""
Implements baseline grid and random search for AutoML.
Supports any sklearn-compatible estimator or wrapped model.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, Literal, Any
import numpy as np

SearchType = Literal["grid", "random"]

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

    def run(self, pipeline: Pipeline, X, y):
        """Run grid or random search on given pipeline."""
        if self.search_type == "grid":
            self.search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
            )
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
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        self.search.fit(X, y)
        return self.search.best_estimator_, self.search.best_params_, self.search.best_score_
