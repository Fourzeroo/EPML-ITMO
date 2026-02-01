"""Run 15+ experiments with different algorithms and hyperparameters."""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from titanic_classifier.config import RANDOM_STATE, RAW_DATA_PATH, TEST_SIZE
from titanic_classifier.mlflow_utils import (
    experiment_context,
    log_model_metrics,
    setup_mlflow,
)


def load_and_prepare_data():
    """Load and prepare Titanic data."""
    df = pd.read_csv(RAW_DATA_PATH)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]].dropna()

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_experiment(name: str, model, X_train, X_test, y_train, y_test, scale=False):
    """Run single experiment with logging."""

    with experiment_context(name, tags={"model_type": type(model).__name__}) as _run:
        # Scale if needed
        if scale:
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
        else:
            X_train_proc = X_train
            X_test_proc = X_test

        # Log parameters
        mlflow.log_params(model.get_params())
        mlflow.log_param("scaling", scale)

        # Train
        model.fit(X_train_proc, y_train)

        # Predict
        y_pred = model.predict(X_test_proc)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_proc)[:, 1]

        # Log metrics
        metrics = log_model_metrics(y_test, y_pred, y_proba)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

        return metrics


def main():
    """Run all experiments."""
    setup_mlflow("titanic-experiments")

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print("\n" + "="*60)
    print("Running 15+ experiments...")
    print("="*60 + "\n")

    experiments = [
        # Random Forest variations (5 experiments)
        ("RF_n50_d5", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE)),
        ("RF_n100_d5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)),
        ("RF_n100_d10", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        ("RF_n200_d10", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)),
        ("RF_n200_d15", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE)),

        # Gradient Boosting variations (3 experiments)
        ("GB_n50_lr01", GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=RANDOM_STATE)),
        ("GB_n100_lr01", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)),
        ("GB_n100_lr05", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)),

        # Logistic Regression variations (2 experiments)
        ("LR_C1", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
        ("LR_C01", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)),

        # Decision Tree variations (2 experiments)
        ("DT_d5", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("DT_d10", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),

        # KNN variations (2 experiments)
        ("KNN_k3", KNeighborsClassifier(n_neighbors=3)),
        ("KNN_k5", KNeighborsClassifier(n_neighbors=5)),

        # AdaBoost (1 experiment)
        ("AdaBoost_n50", AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    ]

    # Models that need scaling
    scale_models = {"LR_C1", "LR_C01", "KNN_k3", "KNN_k5"}

    results = []
    for name, model in experiments:
        scale = name in scale_models
        metrics = run_experiment(name, model, X_train, X_test, y_train, y_test, scale=scale)
        results.append({"name": name, **metrics})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Top 5 models by F1-score")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print(results_df.head().to_string(index=False))

    print(f"\nTotal experiments: {len(experiments)}")
    print("\nDone! Check MLflow UI at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
