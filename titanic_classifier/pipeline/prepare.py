"""Data preparation pipeline stage."""

from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml


def load_config(config_path: str = "configs/data/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> pd.DataFrame:
    """Load and prepare Titanic data."""
    raw_path = config["raw_path"]
    features = config["features"]
    target = config["target"]

    logger.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)

    # Select features and target
    columns = features + [target]
    df = df[columns]

    logger.info(f"Original shape: {df.shape}")

    # Drop rows with missing values
    df = df.dropna()
    logger.info(f"Shape after dropping NaN: {df.shape}")

    # Encode categorical features
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

    logger.info("Categorical features encoded")

    return df


def main():
    """Main function for data preparation."""
    config = load_config()

    # Prepare data
    df = prepare_data(config)

    # Save processed data
    output_path = Path(config["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.success(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
