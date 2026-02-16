from clearml import PipelineController, Task


def create_pipeline():
    pipe = PipelineController(
        name="Titanic-ML-Pipeline",
        project="Titanic-Classification",
        version="1.0",
        add_pipeline_tags=True,
    )

    pipe.add_function_step(
        name="prepare_data",
        function=prepare_data,
        function_return=["X_train", "X_test", "y_train", "y_test"],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name="train_model",
        function=train_model,
        function_kwargs={
            "X_train": "${prepare_data.X_train}",
            "X_test": "${prepare_data.X_test}",
            "y_train": "${prepare_data.y_train}",
            "y_test": "${prepare_data.y_test}",
        },
        function_return=["model", "model_path"],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name="evaluate_model",
        function=evaluate_model,
        function_kwargs={
            "model": "${train_model.model}",
            "X_test": "${prepare_data.X_test}",
            "y_test": "${prepare_data.y_test}",
        },
        function_return=["metrics"],
        cache_executed_step=True,
    )

    pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline completed! Check ClearML UI for results.")


def prepare_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    print("Loading and preparing data...")

    df = pd.read_csv("data/raw/train.csv")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    df = df[features + ["Survived"]].dropna()

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Data prepared: train={len(X_train)}, test={len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    from pathlib import Path

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    print("Training RandomForest model...")

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    model_path = Path("models/pipeline_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    return model, str(model_path)


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    task = Task.init(
        project_name="Titanic-Classification",
        task_name="Pipeline-Controller",
        task_type=Task.TaskTypes.controller,
    )

    create_pipeline()

    task.close()
