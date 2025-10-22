"""
Defines default parameter grids for supported models.
"""

def get_param_grid(model_name: str):
    grids = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "ridge": {
            "model__alpha": [0.1, 1.0, 10.0],
        },
        "decision_tree": {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10],
        },
        "random_forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        },
        "neural_net": {
            "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }
    return grids.get(model_name, {})
