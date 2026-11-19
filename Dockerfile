FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/runtime/uv-cache VIRTUAL_ENV=/runtime/venv \
    PATH=/runtime/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY container/ ./container/
ENTRYPOINT ["sh", "/app/container/entrypoint.sh"]
CMD ["python", "-m", "src.runner", "--plan", "experiments/sample.json"]
