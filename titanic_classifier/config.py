"""Configuration for Titanic Classifier project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

# MLflow configuration
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_DIR / "artifacts")
MLFLOW_EXPERIMENT_NAME = "titanic-classification"

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "train.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "train_processed.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
