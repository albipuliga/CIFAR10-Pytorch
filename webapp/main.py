"""FastAPI entrypoint for CIFAR-10 inference web app."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.api.routes import router
from webapp.core.config import settings
from webapp.services.model_registry import ModelRegistry

logger = logging.getLogger("webapp")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()

    model_registry = ModelRegistry(settings)
    model_registry.load_all()

    app.state.settings = settings
    app.state.model_registry = model_registry
    app.state.templates = Jinja2Templates(directory=str(settings.templates_dir))

    logger.info(
        json.dumps(
            {
                "event": "startup",
                "models_loaded": [model_id.value for model_id in model_registry.loaded_model_ids],
                "checkpoints_dir": str(settings.checkpoints_dir),
            }
        )
    )

    yield
    logger.info(json.dumps({"event": "shutdown"}))


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
if settings.reports_dir.exists():
    app.mount(
        "/reports-assets",
        StaticFiles(directory=str(settings.reports_dir)),
        name="reports-assets",
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id

    start = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - start) * 1000

    logger.info(
        json.dumps(
            {
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{elapsed_ms:.2f}"
    return response


app.include_router(router)
