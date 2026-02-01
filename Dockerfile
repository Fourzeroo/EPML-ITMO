# =====================================================
# Titanic Classifier - Production Docker Image
# =====================================================
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* LICENSE ./

RUN poetry install --no-root --only main || \
    poetry install --no-root --only main || \
    poetry install --no-root --only main

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

COPY --chown=appuser:appgroup titanic_classifier/ ./titanic_classifier/
COPY --chown=appuser:appgroup pyproject.toml ./

RUN mkdir -p data/raw data/processed models logs \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["python", "-m", "titanic_classifier"]

LABEL maintainer="Vladimir" \
      version="0.0.1" \
      description="Titanic survival prediction classifier"
