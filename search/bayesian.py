import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score
import numpy as np

class BayesianOptimizer:
    def __init__(self, model_name, model_obj, param_space, scoring="accuracy", cv=3):
        self.model_name = model_name
        self.model_obj = model_obj
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.best_params = None
        self.best_score = None
        self.best_estimator = None

    def _objective(self, trial, pipe, X, y):
        # Build params from param space
        params = {}
        for pname, spec in self.param_space.items():
            if spec["type"] == "int":
                params[pname] = trial.suggest_int(pname, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[pname] = trial.suggest_float(pname, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])

        pipe.set_params(**params)

        scores = cross_val_score(pipe, X, y, cv=self.cv, scoring=self.scoring)
        return np.mean(scores)

    def run(self, pipe, X, y, n_trials=12):
        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))

        study.optimize(lambda trial: self._objective(trial, pipe, X, y), n_trials=n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        # Retrain best estimator
        pipe.set_params(**self.best_params)
        pipe.fit(X, y)
        self.best_estimator = pipe

        # Build trial history for plotting & saving
        bayes_history = []
        for t in study.trials:
            bayes_history.append({
                "iteration": t.number + 1,
                "value": t.value,
                "params": t.params
            })

        return self.best_estimator, self.best_params, self.best_score, bayes_history
