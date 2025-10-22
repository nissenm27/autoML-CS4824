from sklearn.linear_model import LogisticRegression, Ridge
from .base_model import BaseModel

# logistic regression implementation using sklearn
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(task_type="classification")
        self.model = LogisticRegression(max_iter=1000)
# ridge regression implementation using sklearn
class RidgeRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(task_type="regression")
        self.model = Ridge(alpha=1.0)