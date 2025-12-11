import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

class TinyNAS:
    """
    Small-scale Neural Architecture Search (proof of concept)
    Searches over: number of layers, units per layer, activation, learning rate.
    """

    def __init__(self, cv=3, max_evals=12):
        self.task_type = "classification"
        self.cv = cv
        self.max_evals = max_evals
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf

    def search(self, X, y):
        layer_choices = [(32,), (64,), (32,16), (64,32)]
        activations = ["relu", "tanh"]
        learning_rates = [1e-3, 3e-3, 1e-2]

        # Build search space
        space = list(itertools.product(layer_choices, activations, learning_rates))
        np.random.shuffle(space)

        for i, (layers, act, lr) in enumerate(space[:self.max_evals]):
            model = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=act,
                learning_rate_init=lr,
                max_iter=300
            )

            score = np.mean(cross_val_score(model, X, y, cv=self.cv, scoring="accuracy"))

            if score > self.best_score:
                self.best_score = score
                self.best_params = {
                    "hidden_layer_sizes": layers,
                    "activation": act,
                    "learning_rate_init": lr
                }
                self.best_model = model

        # Fit best model on full data
        self.best_model.fit(X, y)
        return self.best_model, self.best_params, self.best_score
