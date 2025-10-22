# autoML-CS4824/models/tree_models.py
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .base_model import BaseModel

# Simple decision tree using sklearn
class DecisionTreeModel(BaseModel):
    def __init__(self, task_type="classification"):
        super().__init__(task_type)
        if task_type == "classification":
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            self.model = DecisionTreeRegressor(random_state=42)

# Basic random forest implementation using sklearn
class RandomForestModel(BaseModel):
    def __init__(self, task_type="classification"):
        super().__init__(task_type)
        if task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )

# basic gradient boosting model using sklearn
class GradientBoostingModel(BaseModel):
    def __init__(self, task_type="classification"):
        super().__init__(task_type)
        if task_type == "classification":
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
