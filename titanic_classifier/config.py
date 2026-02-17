from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_DIR / "artifacts")
MLFLOW_EXPERIMENT_NAME = "titanic-classification"

RAW_DATA_PATH = DATA_DIR / "raw" / "train.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "train_processed.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
