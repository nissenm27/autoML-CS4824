"""
Defines parameter grids and Bayesian search spaces for supported models.
Now supports META-LEARNING warm starts.
"""

def get_param_grid(model_name: str, search_type: str = "grid", warm_start=None):

    # ==============================
    # BASELINE GRID SEARCH SPACES
    # ==============================
    grid_spaces = {
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

    # ==============================
    # BASELINE BAYESIAN SPACES
    # ==============================
    bayes_spaces = {
        "logistic_regression": {
            "model__C": {"type": "float", "low": 1e-3, "high": 10, "log": True},
            "model__solver": {"type": "categorical", "choices": ["lbfgs", "liblinear"]},
        },
        "ridge": {
            "model__alpha": {"type": "float", "low": 1e-3, "high": 10, "log": True},
        },
        "decision_tree": {
            "model__max_depth": {"type": "int", "low": 2, "high": 20},
            "model__min_samples_split": {"type": "int", "low": 2, "high": 10},
        },
        "random_forest": {
            "model__n_estimators": {"type": "int", "low": 50, "high": 300},
            "model__max_depth": {"type": "int", "low": 2, "high": 20},
        },
        "gradient_boosting": {
            "model__n_estimators": {"type": "int", "low": 50, "high": 300},
            "model__learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
            "model__max_depth": {"type": "int", "low": 2, "high": 10},
        },
        "neural_net": {
            "model__hidden_layer_sizes": {
                "type": "categorical",
                "choices": [(32,), (64,), (128,), (64, 32)]
            },
            "model__learning_rate_init": {
                "type": "float", "low": 1e-4, "high": 1e-1, "log": True
            },
        },
    }

    # ==============================================
    # APPLY META-LEARNING WARM START (BAYES ONLY)
    # ==============================================
    if search_type == "bayes" and warm_start:
        space = bayes_spaces[model_name].copy()

        # ---- n_estimators warm start ----
        if "n_estimators" in warm_start:
            base = warm_start["n_estimators"]
            space["model__n_estimators"] = {
                "type": "int",
                "low": max(10, base - 50),
                "high": base + 50
            }

        # ---- max_depth warm start ----
        if "max_depth" in warm_start:
            base = warm_start["max_depth"]
            space["model__max_depth"] = {
                "type": "int",
                "low": max(1, base - 2),
                "high": base + 2
            }

        # ---- learning_rate warm start ----
        if "learning_rate" in warm_start:
            base = warm_start["learning_rate"]
            space["model__learning_rate"] = {
                "type": "float",
                "low": max(1e-4, base / 3),
                "high": min(1.0, base * 3),
                "log": True
            }

        # ---- neural_net warm starts ----
        if "hidden_layer_sizes" in warm_start:
            hls = warm_start["hidden_layer_sizes"]
            space["model__hidden_layer_sizes"] = {
                "type": "categorical",
                "choices": [hls]  # bias search toward known good architecture
            }

        if "learning_rate_init" in warm_start:
            base = warm_start["learning_rate_init"]
            space["model__learning_rate_init"] = {
                "type": "float",
                "low": max(1e-5, base / 5),
                "high": base * 5,
                "log": True
            }

        return space

    # ==============================================
    # FALLBACK: Standard Search Spaces
    # ==============================================
    if search_type == "bayes":
        return bayes_spaces.get(model_name, {})

    return grid_spaces.get(model_name, {})
