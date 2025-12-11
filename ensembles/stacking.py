import numpy as np
from sklearn.linear_model import LogisticRegression

class StackingEnsemble:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=200)

    def fit(self, X, y):
        # Level-1 features = concatenated predicted probabilities
        base_preds = [model.predict_proba(X) for model in self.base_models]
        base_preds = np.hstack(base_preds)
        self.meta_model.fit(base_preds, y)

    def predict(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models]
        base_preds = np.hstack(base_preds)
        return self.meta_model.predict(base_preds)

    def predict_proba(self, X):
        base_preds = [model.predict_proba(X) for model in self.base_models]
        base_preds = np.hstack(base_preds)
        return self.meta_model.predict_proba(base_preds)
