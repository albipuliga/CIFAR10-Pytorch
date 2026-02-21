"""Typed API schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelId(str, Enum):
    baseline = "baseline"
    cnnv2 = "cnnv2"


class TopKPrediction(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    model_id: ModelId
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKPrediction]
    inference_ms: float = Field(ge=0.0)
    request_id: str | None = None


class ReportFigure(BaseModel):
    name: str
    url: str


class ReportSummaryResponse(BaseModel):
    metrics: dict[str, Any]
    figures: list[ReportFigure]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class ModelMetadata(BaseModel):
    id: ModelId
    checkpoint: str
    classes_count: int


class ModelsResponse(BaseModel):
    models: list[ModelMetadata]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[ModelId]
    version: str
