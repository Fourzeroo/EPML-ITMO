# Titanic Classifier

ML проект для предсказания выживания пассажиров Титаника с полным MLOps workflow.

## Возможности

- **Версионирование данных** — DVC для контроля версий датасетов
- **Трекинг экспериментов** — MLflow и ClearML для логирования
- **Автоматизация** — DVC Pipelines + Hydra для конфигураций
- **Качество кода** — Pre-commit hooks, Ruff, MyPy

## Быстрый старт
```bash
git clone https://github.com/Fourzeroo/EPML-ITMO.git
cd EPML-ITMO
poetry install
poetry run dvc pull
poetry run dvc repro
```

## Технологии

| Категория | Инструменты |
|-----------|-------------|
| ML | scikit-learn, pandas, numpy |
| MLOps | DVC, MLflow, ClearML, Hydra |
| Качество кода | Ruff, MyPy, Bandit, pre-commit |
| Документация | MkDocs Material |
