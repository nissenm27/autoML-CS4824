from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from typing import Literal

# Unified API with train, test, score so the AutoML system can loop over models easily
class BaseModel:
    def __init__(self, task_type: Literal["classification", "regression"]):
        self.task_type = task_type
        self.model = None

    # Fit specific model
    def train(self, X, y):
        self.model.fit(X, y)

    # Train specific model
    def predict(self, X):
        return self.model.predict(X)

    # Evaluate specific model on appropriate metrics
    def score(self, X, y):
        preds = self.predict(X)
        if self.task_type == "classification":
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="weighted")
            return {"accuracy": acc, "f1": f1}
        else:
            rmse = mean_squared_error(y, preds, squared=False)
            return {"rmse": rmse}
