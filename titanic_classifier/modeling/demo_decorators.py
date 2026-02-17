import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from titanic_classifier.config import RANDOM_STATE, RAW_DATA_PATH
from titanic_classifier.mlflow_utils import (
    experiment_context,
    get_best_run,
    log_experiment,
    log_model_metrics,
    setup_mlflow,
)


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[features + ["Survived"]].dropna()

    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    X = df[features]
    y = df["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# Пример 1: декоратор @log_experiment
@log_experiment(run_name="decorator_example", tags={"demo": "decorator"})
def train_with_decorator():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    mlflow.log_params(model.get_params())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = log_model_metrics(y_test, y_pred, y_proba)
    mlflow.sklearn.log_model(model, "model")

    print(f"[Decorator] Accuracy: {metrics['accuracy']:.4f}")
    return metrics


# Пример 2: контекстный менеджер
def train_with_context_manager():
    X_train, X_test, y_train, y_test = load_data()

    params = {"n_estimators": 150, "max_depth": 8}

    with experiment_context("context_manager_example", params=params, tags={"demo": "context_manager"}):
        model = RandomForestClassifier(**params, random_state=RANDOM_STATE)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = log_model_metrics(y_test, y_pred, y_proba)
        mlflow.sklearn.log_model(model, "model")

        print(f"[Context Manager] Accuracy: {metrics['accuracy']:.4f}")


# Пример 3: получение лучшего запуска
def show_best_run():
    setup_mlflow("titanic-experiments")

    best = get_best_run(metric="f1_score")
    if best:
        print("\n" + "="*50)
        print("BEST RUN (by F1-score):")
        print("="*50)
        print(f"Run ID: {best.get('run_id', 'N/A')}")
        print(f"F1-score: {best.get('metrics.f1_score', 'N/A'):.4f}")
        print(f"Accuracy: {best.get('metrics.accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    setup_mlflow("titanic-experiments")

    print("="*50)
    print("Demo: Decorators and Context Managers")
    print("="*50 + "\n")

    print("1. Training with @log_experiment decorator...")
    train_with_decorator()

    print("\n2. Training with context manager...")
    train_with_context_manager()

    print("\n3. Getting best run...")
    show_best_run()

    print("\n" + "="*50)
    print("Done! Check MLflow UI for new experiments")
    print("="*50)
