"""Model training pipeline stage."""

from pathlib import Path
import pickle  # nosec B403
import sys

from loguru import logger
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml


def parse_args() -> dict:
    """Parse command line arguments in key=value format."""
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[key] = value
    return args


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(model_config: dict):
    """Create model instance from config."""
    model_name = model_config.get("name", "RandomForest")

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 100),
            max_depth=model_config.get("max_depth", 10),
            min_samples_split=model_config.get("min_samples_split", 2),
            min_samples_leaf=model_config.get("min_samples_leaf", 1),
            random_state=42,
        )
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=model_config.get("n_estimators", 100),
            max_depth=model_config.get("max_depth", 3),
            learning_rate=model_config.get("learning_rate", 0.1),
            random_state=42,
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main function for model training."""
    # Parse command line args
    args = parse_args()
    model_name = args.get("model", "random_forest")

    # Load configs
    data_config = load_config("configs/data/default.yaml")
    train_config = load_config("configs/train/default.yaml")
    model_config = load_config(f"configs/model/{model_name}.yaml")

    # Load processed data
    data_path = data_config["processed_path"]
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Split features and target
    features = data_config["features"]
    target = data_config["target"]

    X = df[features]
    y = df[target]

    # Train/test split
    test_size = data_config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Create and train model
    model = get_model(model_config)
    logger.info(f"Training {model_config['name']}...")
    model.fit(X_train, y_train)

    # Save model
    model_path = Path(train_config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"Model saved to {model_path}")

    # Save test data for evaluation
    test_data_dir = Path("data/processed")
    X_test.to_csv(test_data_dir / "X_test.csv", index=False)
    y_test.to_csv(test_data_dir / "y_test.csv", index=False)
    logger.info("Test data saved for evaluation")


if __name__ == "__main__":
    main()
