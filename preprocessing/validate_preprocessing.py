import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .schema import infer_schema
from .build_preprocessor import build_preprocessor
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    df = pd.read_csv("data/adult_income.csv")
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]

    schema = infer_schema(X)
    preprocessor = build_preprocessor(schema)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.3f}")
