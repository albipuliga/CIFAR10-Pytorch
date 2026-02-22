FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app
# Install all locked deps except PyTorch binaries, then install CPU-only wheels
# to avoid pulling large CUDA packages during cloud builds.
RUN uv sync --frozen --no-dev --no-install-project --no-install-package torch --no-install-package torchvision \
    && uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu" "torchvision==0.25.0+cpu"

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
