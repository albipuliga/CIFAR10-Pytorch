"""Application and API routes."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Annotated
from urllib.parse import quote

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from webapp.core.config import Settings
from webapp.core.constants import CIFAR10_CLASSES
from webapp.schemas.prediction import (
    ErrorResponse,
    HealthResponse,
    ModelId,
    ModelsResponse,
    PredictionResponse,
    ReportFigure,
    ReportSummaryResponse,
    TopKPrediction,
)
from webapp.services.model_registry import ModelRegistry
from webapp.services.preprocess import (
    InvalidImageError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
    image_bytes_to_tensor,
    validate_upload,
)

router = APIRouter()


@torch.inference_mode()
def _predict_with_model(
    *,
    registry: ModelRegistry,
    model_id: ModelId,
    image_tensor: torch.Tensor,
    top_k: int,
    request_id: str | None,
) -> PredictionResponse:
    model = registry.get_model(model_id)
    start = perf_counter()
    logits = model(image_tensor.to(registry.device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    inference_ms = (perf_counter() - start) * 1000

    safe_top_k = max(1, min(top_k, len(CIFAR10_CLASSES)))
    values, indices = torch.topk(probabilities, safe_top_k)
    predictions = [
        TopKPrediction(
            class_name=CIFAR10_CLASSES[index],
            probability=round(float(probability), 6),
        )
        for probability, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]

    return PredictionResponse(
        model_id=model_id,
        predicted_class=predictions[0].class_name,
        confidence=predictions[0].probability,
        top_k=predictions,
        inference_ms=round(float(inference_ms), 3),
        request_id=request_id,
    )


def _load_report_metrics(settings: Settings) -> dict[str, object]:
    reports_dir = settings.reports_dir
    expected_path = reports_dir / "results.json"
    if not reports_dir.exists():
        return {
            "status": "missing",
            "message": f"No report metrics found at {expected_path}.",
        }

    metrics_path = expected_path
    if not metrics_path.exists():
        json_candidates = sorted(
            path
            for path in reports_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        if not json_candidates:
            return {
                "status": "missing",
                "message": f"No report metrics found at {expected_path}.",
            }
        metrics_path = json_candidates[0]

    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {
            "status": "invalid",
            "message": f"{metrics_path.name} is not valid JSON.",
        }

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"models": data}
    return {
        "status": "invalid",
        "message": f"{metrics_path.name} is not a JSON object or array.",
    }


def _load_report_figures(settings: Settings) -> list[ReportFigure]:
    if not settings.reports_dir.exists():
        return []

    figures: list[ReportFigure] = []
    figure_extensions = {".png", ".jpg", ".jpeg", ".svg", ".webp"}

    # Top-level report dir
    for path in sorted(settings.reports_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in figure_extensions:
            continue
        figures.append(
            ReportFigure(
                name=path.stem.replace("_", " ").title(),
                url=f"/reports-assets/{quote(path.name)}",
            )
        )

    # figures/ subdirectory
    figures_dir = settings.reports_dir / "figures"
    if figures_dir.is_dir():
        for path in sorted(figures_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in figure_extensions:
                continue
            figures.append(
                ReportFigure(
                    name=path.stem.replace("_", " ").title(),
                    url=f"/reports-assets/figures/{quote(path.name)}",
                )
            )

    return figures


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    template = request.app.state.templates
    return template.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_model": settings.default_model_id,
            "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    registry = request.app.state.model_registry
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded_model_ids,
        version=settings.version,
    )


@router.get("/api/v1/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    registry = request.app.state.model_registry
    return ModelsResponse(models=registry.list_model_metadata())


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
    },
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: Annotated[ModelId, Form()] = ModelId.cnnv2,
    top_k: Annotated[int, Form(ge=1, le=10)] = 5,
) -> PredictionResponse:
    settings = request.app.state.settings

    raw_bytes = await file.read()
    try:
        validate_upload(
            content_type=file.content_type,
            raw_bytes=raw_bytes,
            max_upload_bytes=settings.max_upload_bytes,
        )
        image_tensor = image_bytes_to_tensor(
            image_bytes=raw_bytes,
            mean=settings.normalization_mean,
            std=settings.normalization_std,
        )
    except UnsupportedMediaTypeError as exc:
        raise HTTPException(
            status_code=415,
            detail=str(exc),
        ) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    registry = request.app.state.model_registry
    request_id = getattr(request.state, "request_id", None)
    return _predict_with_model(
        registry=registry,
        model_id=model_id,
        image_tensor=image_tensor,
        top_k=top_k,
        request_id=request_id,
    )


@router.get("/api/v1/reports", response_model=ReportSummaryResponse)
async def reports(request: Request) -> ReportSummaryResponse:
    settings = request.app.state.settings
    return ReportSummaryResponse(
        metrics=_load_report_metrics(settings),
        figures=_load_report_figures(settings),
    )
