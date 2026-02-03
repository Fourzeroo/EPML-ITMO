# Развертывание

## Docker
```bash
docker build -t titanic-classifier:latest .
docker run --rm -v $(pwd)/data:/app/data titanic-classifier:latest
```

## Воспроизведение
```bash
git clone https://github.com/Fourzeroo/EPML-ITMO.git
cd EPML-ITMO
poetry install
poetry run dvc pull
poetry run dvc repro
```
