import numpy as np

class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models  # list of fitted sklearn estimators

    def predict_proba(self, X):
        # Average predicted probabilities
        probs = [model.predict_proba(X) for model in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
