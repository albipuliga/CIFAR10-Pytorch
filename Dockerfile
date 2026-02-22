FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

WORKDIR /app

COPY requirements.railway.txt /app/requirements.railway.txt
COPY requirements.railway.torch.txt /app/requirements.railway.torch.txt
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
    --index-url https://pypi.org/simple \
    -r /app/requirements.railway.txt \
    && uv pip install --python /opt/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.railway.torch.txt

FROM python:3.14-slim AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY webapp /app/webapp
COPY src/checkpoints /app/src/checkpoints
COPY src/reports /app/src/reports

EXPOSE 8000

FROM app-base AS dev
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
RUN uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple watchfiles

CMD ["sh", "-c", "uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM app-base AS runtime
CMD ["sh", "-c", "uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
