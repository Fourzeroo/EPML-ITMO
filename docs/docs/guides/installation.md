# Установка

Подробные инструкции по настройке окружения разработки.

## Требования

- Python 3.10+
- Poetry 1.5+
- Git

## Poetry (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/Fourzeroo/EPML-ITMO.git
cd EPML-ITMO

# Установка зависимостей
poetry install

# Активация виртуального окружения
poetry shell

# Установка pre-commit хуков
pre-commit install

# Загрузка данных из DVC
dvc pull
```

## Docker

```bash
# Сборка образа
docker build -t titanic-classifier:latest .

# Запуск контейнера
docker run --rm titanic-classifier:latest

# Запуск с монтированием данных
docker run --rm -v $(pwd)/data:/app/data titanic-classifier:latest
```

## Проверка установки

```bash
# Запуск пайплайна
poetry run dvc repro

# Проверка метрик
cat reports/metrics.json

# Запуск тестов
poetry run pytest
```

## Настройка MLflow

MLflow настраивается автоматически при первом запуске. Для просмотра экспериментов:

```bash
poetry run mlflow ui
```

Откройте http://localhost:5000 в браузере.
