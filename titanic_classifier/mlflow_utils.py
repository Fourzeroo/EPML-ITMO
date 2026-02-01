"""MLflow utilities: decorators, context managers, and helper functions."""

from contextlib import contextmanager
import functools
import time
from typing import Any, Callable, Dict, Optional

from loguru import logger
import mlflow
import pandas as pd

from titanic_classifier.config import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def setup_mlflow(experiment_name: Optional[str] = None) -> None:
    """Initialize MLflow with configured settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    exp_name = experiment_name or MLFLOW_EXPERIMENT_NAME

    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        mlflow.create_experiment(
            exp_name,
            artifact_location=MLFLOW_ARTIFACT_LOCATION,
        )
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow configured: experiment='{exp_name}'")


def log_experiment(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable:
    """Decorator for automatic MLflow logging.

    Usage:
        @log_experiment(run_name="my_experiment")
        def train_model(params):
            ...
            return metrics
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            setup_mlflow()

            with mlflow.start_run(run_name=run_name):
                if tags:
                    mlflow.set_tags(tags)

                # Log start time
                start_time = time.time()

                # Execute function
                result = func(*args, **kwargs)

                # Log duration
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)

                logger.info(f"Experiment '{run_name}' completed in {duration:.2f}s")

                return result

        return wrapper

    return decorator


@contextmanager
def experiment_context(
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Context manager for MLflow experiments.

    Usage:
        with experiment_context("my_run", params={"lr": 0.01}) as run:
            model.fit(X, y)
            mlflow.log_metric("accuracy", 0.95)
    """
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        start_time = time.time()
        logger.info(f"Starting experiment: {run_name}")

        yield run

        duration = time.time() - start_time
        mlflow.log_metric("duration_seconds", duration)
        logger.info(f"Experiment '{run_name}' completed in {duration:.2f}s")


def log_model_metrics(
    y_true,
    y_pred,
    y_proba=None,
    prefix: str = "",
) -> Dict[str, float]:
    """Log classification metrics to MLflow."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # Add prefix and log
    for name, value in metrics.items():
        metric_name = f"{prefix}_{name}" if prefix else name
        mlflow.log_metric(metric_name, value)

    return metrics


def get_best_run(
    experiment_name: Optional[str] = None,
    metric: str = "f1_score",
    ascending: bool = False,
) -> Optional[Dict]:
    """Get the best run from an experiment."""
    setup_mlflow(experiment_name)

    experiment = mlflow.get_experiment_by_name(
        experiment_name or MLFLOW_EXPERIMENT_NAME
    )
    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
        output_format="pandas",
    )

    if not isinstance(runs, pd.DataFrame) or runs.empty:
        return None

    return runs.iloc[0].to_dict()
