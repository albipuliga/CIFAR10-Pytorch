"""Typed API schemas."""

from __future__ import annotations

from enum import Enum

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


class ReportMetricEntry(BaseModel):
    model: ModelId
    test_accuracy: float | None = None
    test_precision_macro: float | None = None
    test_recall_macro: float | None = None
    test_f1_macro: float | None = None


class ReportMetrics(BaseModel):
    models: list[ReportMetricEntry] = Field(default_factory=list)


class ReportSummaryResponse(BaseModel):
    metrics: ReportMetrics
    figures: list[ReportFigure]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[ModelId]
    version: str
