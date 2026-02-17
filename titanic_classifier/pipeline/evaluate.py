import json
from pathlib import Path
import pickle  # nosec B403

from hydra import compose, initialize_config_dir
from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main():
    config_dir = str(Path.cwd() / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    model_path = cfg.train.model_path
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)  # nosec B301

    logger.info("Loading test data")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
