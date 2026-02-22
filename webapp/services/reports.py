"""Report artifact loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from webapp.core.config import Settings
from webapp.schemas.prediction import (
    ModelId,
    ReportFigure,
    ReportMetricEntry,
    ReportMetrics,
    ReportSummaryResponse,
)

_METRICS_FILENAME = "results.json"
_CONFUSION_MATRICES = {
    ModelId.baseline: "figures/confusion_matrix_baseline.png",
    ModelId.cnnv2: "figures/confusion_matrix_cnnv2.png",
}


def _to_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_metrics(metrics_path: Path) -> ReportMetrics:
    if not metrics_path.exists():
        return ReportMetrics()

    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return ReportMetrics()

    if not isinstance(data, list):
        return ReportMetrics()

    rows: list[ReportMetricEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        raw_model_id = item.get("model")
        try:
            model_id = ModelId(raw_model_id)
        except ValueError:
            continue

        rows.append(
            ReportMetricEntry(
                model=model_id,
                test_accuracy=_to_optional_float(item.get("test_accuracy")),
                test_precision_macro=_to_optional_float(item.get("test_precision_macro")),
                test_recall_macro=_to_optional_float(item.get("test_recall_macro")),
                test_f1_macro=_to_optional_float(item.get("test_f1_macro")),
            )
        )

    return ReportMetrics(models=rows)


def _load_confusion_matrices(reports_dir: Path) -> list[ReportFigure]:
    figures: list[ReportFigure] = []
    for model_id, relative_path in _CONFUSION_MATRICES.items():
        figure_path = reports_dir / relative_path
        if not figure_path.exists():
            continue
        figures.append(
            ReportFigure(
                name=f"Confusion Matrix {model_id.value.upper()}",
                url=f"/reports-assets/{relative_path}",
            )
        )
    return figures


def load_report_summary(settings: Settings) -> ReportSummaryResponse:
    reports_dir = settings.reports_dir
    metrics = _load_metrics(reports_dir / _METRICS_FILENAME)
    figures = _load_confusion_matrices(reports_dir)
    return ReportSummaryResponse(metrics=metrics, figures=figures)
