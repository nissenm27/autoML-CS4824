import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.schema import infer_schema
from preprocessing.build_preprocessor import build_preprocessor
from sklearn.pipeline import Pipeline

from models.linear_models import LogisticRegressionModel
from models.tree_models import DecisionTreeModel, RandomForestModel, GradientBoostingModel
from models.neural_net import NeuralNetworkModel

def test_model(model_class, df_path, target_col):
    df = pd.read_csv(df_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    schema = infer_schema(X)
    pre = build_preprocessor(schema)
    model = model_class()

    pipe = Pipeline([
        ("prep", pre),
        ("model", model.model)  # use sklearn estimator inside wrapper
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"{model_class.__name__}: Score = {score:.3f}")

if __name__ == "__main__":
    test_model(LogisticRegressionModel, "data/adult_income.csv", "class")
    test_model(DecisionTreeModel, "data/adult_income.csv", "class")
    test_model(RandomForestModel, "data/adult_income.csv", "class")
    test_model(GradientBoostingModel, "data/adult_income.csv", "class")
    test_model(NeuralNetworkModel, "data/adult_income.csv", "class")
