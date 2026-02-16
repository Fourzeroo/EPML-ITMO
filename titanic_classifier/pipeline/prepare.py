from pathlib import Path

from hydra import compose, initialize_config_dir
from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    config_dir = str(Path.cwd() / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    raw_path = cfg.data.raw_path
    logger.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Original shape: {df.shape}")

    features = list(cfg.data.features)
    target = cfg.data.target
    columns = features + [target]
    df = df[columns].dropna()
    logger.info(f"Shape after dropping NaN: {df.shape}")

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"])
    logger.info("Categorical features encoded")

    output_path = Path(cfg.data.processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
