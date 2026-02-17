from pathlib import Path
import pickle  # nosec B403
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split


def get_overrides():
    overrides = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
    return overrides


def main():
    config_dir = str(Path.cwd() / "configs")
    overrides = get_overrides()

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    model_name = cfg.model._target_.split(".")[-1]
    logger.info(f"Config loaded: model={model_name}")

    data_path = cfg.data.processed_path
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    features = list(cfg.data.features)
    target = cfg.data.target
    X = df[features]
    y = df[target]

    test_size = cfg.data.test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=cfg.random_state
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = instantiate(cfg.model)
    logger.info(f"Training {model_name}...")

    model.fit(X_train, y_train)

    model_path = Path(cfg.train.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"Model saved to {model_path}")

    test_data_dir = Path("data/processed")
    X_test.to_csv(test_data_dir / "X_test.csv", index=False)
    y_test.to_csv(test_data_dir / "y_test.csv", index=False)
    logger.info("Test data saved for evaluation")


if __name__ == "__main__":
    main()
