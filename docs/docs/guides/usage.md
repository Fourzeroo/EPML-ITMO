# Использование

## Запуск пайплайна
```bash
poetry run dvc repro
```

## Обучение с разными моделями (Hydra)
```bash
# RandomForest (по умолчанию)
poetry run python -m titanic_classifier.pipeline.train

# GradientBoosting
poetry run python -m titanic_classifier.pipeline.train model=gradient_boosting

# LogisticRegression
poetry run python -m titanic_classifier.pipeline.train model=logistic_regression
```

## ClearML эксперименты
```bash
poetry run python titanic_classifier/clearml_train.py
```
