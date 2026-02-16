from contextlib import contextmanager
import functools
import time

from loguru import logger
import mlflow
import pandas as pd

from titanic_classifier.config import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def setup_mlflow(experiment_name=None):
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


def log_experiment(run_name=None, tags=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            setup_mlflow()

            with mlflow.start_run(run_name=run_name):
                if tags:
                    mlflow.set_tags(tags)

                start_time = time.time()

                result = func(*args, **kwargs)

                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)

                logger.info(f"Experiment '{run_name}' completed in {duration:.2f}s")

                return result

        return wrapper

    return decorator


@contextmanager
def experiment_context(run_name, params=None, tags=None):
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


def log_model_metrics(y_true, y_pred, y_proba=None, prefix=""):
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

    for name, value in metrics.items():
        metric_name = f"{prefix}_{name}" if prefix else name
        mlflow.log_metric(metric_name, value)

    return metrics


def get_best_run(experiment_name=None, metric="f1_score", ascending=False):
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
