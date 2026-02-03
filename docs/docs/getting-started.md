# Начало работы

Краткое руководство для быстрого запуска проекта.

## Быстрый старт

```bash
# Клонирование и установка
git clone https://github.com/Fourzeroo/EPML-ITMO.git
cd EPML-ITMO
poetry install

# Загрузка данных и запуск пайплайна
poetry run dvc pull
poetry run dvc repro
```

## Что дальше?

- [Установка](guides/installation.md) — детальные инструкции по настройке окружения
- [Использование](guides/usage.md) — запуск экспериментов с разными моделями
- [API](api/models.md) — конфигурации моделей и утилиты
