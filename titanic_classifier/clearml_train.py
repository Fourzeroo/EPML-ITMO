"""Training script with ClearML tracking."""

from pathlib import Path

from clearml import OutputModel, Task
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_and_prepare_data(path: str = "data/raw/train.csv"):
    """Load and prepare Titanic data."""
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


def train_model(model_name: str, model, X_train, X_test, y_train, y_test, scale=False):
    """Train model and log to ClearML."""

    # Initialize ClearML Task
    task = Task.init(
        project_name="Titanic-Classification",
        task_name=model_name,
        task_type=Task.TaskTypes.training,
    )

    # Log hyperparameters
    params = model.get_params()
    task.connect(params, name="hyperparameters")

    # Scale if needed
    if scale:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_proc = X_test.values if hasattr(X_test, 'values') else X_test

    # Train
    model.fit(X_train_proc, y_train)

    # Predict
    y_pred = model.predict(X_test_proc)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_proc)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    # Log metrics to ClearML
    logger = task.get_logger()
    for name, value in metrics.items():
        logger.report_single_value(name=name, value=value)

    # Save model locally
    model_path = Path(f"models/{model_name}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Register model in ClearML
    output_model = OutputModel(task=task, framework="scikit-learn")
    output_model.update_weights(str(model_path))

    print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

    # Close task
    task.close()

    return metrics


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n" + "="*60)
    print("Running experiments with ClearML tracking...")
    print("="*60 + "\n")

    experiments = [
        # Random Forest variations
        ("RF_n50_d5", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42), False),
        ("RF_n100_d10", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), False),
        ("RF_n200_d15", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42), False),

        # Gradient Boosting
        ("GB_n100_lr01", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
        ("GB_n100_lr05", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42), False),

        # Logistic Regression
        ("LR_C1", LogisticRegression(C=1.0, max_iter=1000, random_state=42), True),
        ("LR_C01", LogisticRegression(C=0.1, max_iter=1000, random_state=42), True),

        # Decision Tree
        ("DT_d5", DecisionTreeClassifier(max_depth=5, random_state=42), False),
        ("DT_d10", DecisionTreeClassifier(max_depth=10, random_state=42), False),

        # KNN
        ("KNN_k5", KNeighborsClassifier(n_neighbors=5), True),
    ]

    results = []
    for name, model, scale in experiments:
        metrics = train_model(name, model, X_train, X_test, y_train, y_test, scale)
        results.append({"name": name, **metrics})

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print(results_df.to_string(index=False))
    print(f"\nTotal experiments: {len(experiments)}")
    print("\nCheck ClearML UI at https://app.clear.ml")


if __name__ == "__main__":
    main()
