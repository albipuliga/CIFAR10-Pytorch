FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app
RUN uv sync --frozen --no-dev --no-install-project

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
