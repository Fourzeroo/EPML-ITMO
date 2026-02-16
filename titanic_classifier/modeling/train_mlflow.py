import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(path="data/raw/train.csv"):
    df = pd.read_csv(path)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]].dropna()

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_log(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_params(model.get_params())

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}")

        return accuracy


if __name__ == "__main__":
    mlflow.set_experiment("titanic-classification")  # type: ignore[attr-defined]

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models = [
        ("RandomForest_n50", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("RandomForest_n100", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("RandomForest_n200", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    for name, model in models:
        train_and_log(name, model, X_train, X_test, y_train, y_test)

    print("\nDone! Check MLflow UI at http://127.0.0.1:5000")
