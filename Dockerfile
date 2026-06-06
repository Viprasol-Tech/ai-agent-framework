# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
# Slim image that runs the offline demo / CLI with no API key required.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="ai-agent-framework" \
      org.opencontainers.image.description="Minimal Python framework for building LLM agents." \
      org.opencontainers.image.url="https://viprasol.com" \
      org.opencontainers.image.source="https://github.com/Viprasol-Tech/ai-agent-framework" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY examples ./examples

# Default: run the deterministic, offline demo.
CMD ["python", "examples/demo.py"]
