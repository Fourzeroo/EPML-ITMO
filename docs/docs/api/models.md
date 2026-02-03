# API: Модели

## Конфигурации (Hydra)

Модели в `configs/model/`:
```yaml
# configs/model/random_forest.yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
max_depth: 10
random_state: 42
```

## Доступные модели

- `random_forest` — RandomForestClassifier
- `gradient_boosting` — GradientBoostingClassifier  
- `logistic_regression` — LogisticRegression
