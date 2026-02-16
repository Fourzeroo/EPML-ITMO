FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    POETRY_REQUESTS_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock LICENSE ./

RUN poetry install --no-root --only main

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

COPY --chown=appuser:appgroup titanic_classifier/ ./titanic_classifier/
COPY --chown=appuser:appgroup pyproject.toml ./
COPY --chown=appuser:appgroup data/raw/*.dvc ./data/raw/
COPY --chown=appuser:appgroup .dvc/ ./.dvc/

RUN mkdir -p data/raw data/processed models mlruns logs \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["python", "-m", "titanic_classifier"]
