from sklearn.neural_network import MLPClassifier, MLPRegressor
from .base_model import BaseModel

# Basic MLP neural network using sklearn
class NeuralNetworkModel(BaseModel):
    def __init__(self, task_type="classification"):
        super().__init__(task_type)
        if task_type == "classification":
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)