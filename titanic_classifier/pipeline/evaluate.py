"""Model evaluation pipeline stage."""

import json
from pathlib import Path
import pickle  # nosec B403

from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, "rb") as f:
        return pickle.load(f)  # nosec B301


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model and return metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    return metrics


def main():
    """Main function for model evaluation."""
    # Paths
    model_path = "models/model.pkl"
    X_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"
    metrics_path = Path("reports/metrics.json")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)

    # Load test data
    logger.info("Loading test data")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Evaluate
    logger.info("Evaluating model")
    metrics = evaluate_model(model, X_test, y_test)

    # Print metrics
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    # Save metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
