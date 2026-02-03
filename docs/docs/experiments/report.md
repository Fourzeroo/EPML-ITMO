# Отчёт об экспериментах

## Результаты

Проведено **15+ экспериментов** с различными алгоритмами.

| Алгоритм | Лучший F1 |
|----------|-----------|
| RandomForest | 0.77 |
| GradientBoosting | 0.75 |
| LogisticRegression | 0.71 |

## Лучшая модель

**RandomForest (n_estimators=100, max_depth=10)**

```
Accuracy:  0.83
Precision: 0.79
Recall:    0.71
F1-score:  0.77
```

Конфигурация по умолчанию (`configs/model/random_forest.yaml`):

```yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
random_state: 42
```

## Инструменты

- **MLflow** — локальный трекинг экспериментов
- **ClearML** — облачный трекинг и визуализация
- **Hydra** — управление конфигурациями моделей
