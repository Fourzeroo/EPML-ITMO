# API: Утилиты

## MLflow утилиты

Модуль `titanic_classifier/mlflow_utils.py` предоставляет удобные инструменты для работы с MLflow.

### setup_mlflow

Инициализация MLflow с настройками проекта.

```python
setup_mlflow(experiment_name: Optional[str] = None) -> None
```

**Параметры:**

- `experiment_name` — имя эксперимента (опционально, по умолчанию из конфига)

### @log_experiment

Декоратор для автоматического логирования экспериментов.

```python
@log_experiment(run_name: str = None, tags: Dict[str, str] = None)
def train_model(params):
    ...
    return metrics
```

**Параметры:**

- `run_name` — имя запуска
- `tags` — словарь тегов для эксперимента

### experiment_context

Контекстный менеджер для MLflow экспериментов.

```python
with experiment_context("my_run", params={"lr": 0.01}) as run:
    model.fit(X, y)
    mlflow.log_metric("accuracy", 0.95)
```

**Параметры:**

- `run_name` — имя запуска
- `params` — словарь параметров для логирования
- `tags` — словарь тегов

### log_model_metrics

Логирование метрик классификации в MLflow.

```python
log_model_metrics(y_true, y_pred, y_proba=None, prefix: str = "") -> Dict[str, float]
```

**Параметры:**

- `y_true` — истинные метки
- `y_pred` — предсказанные метки
- `y_proba` — вероятности предсказаний (опционально, для ROC AUC)
- `prefix` — префикс для имён метрик

**Возвращает:** словарь с метриками (accuracy, precision, recall, f1_score, roc_auc)

### get_best_run

Получение лучшего эксперимента по метрике.

```python
get_best_run(
    experiment_name: Optional[str] = None,
    metric: str = "f1_score",
    ascending: bool = False
) -> Optional[Dict]
```

**Параметры:**

- `experiment_name` — имя эксперимента
- `metric` — метрика для сортировки (по умолчанию `f1_score`)
- `ascending` — сортировка по возрастанию

**Возвращает:** словарь с данными лучшего запуска или `None`
