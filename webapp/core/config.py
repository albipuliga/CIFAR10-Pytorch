"""Runtime settings for the FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from webapp.core import constants


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    app_name: str
    version: str
    max_upload_bytes: int
    default_model_id: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    @property
    def checkpoints_dir(self) -> Path:
        return Path(
            os.getenv(
                "CHECKPOINTS_DIR",
                self.repo_root / "src" / "checkpoints",
            )
        )

    @property
    def reports_dir(self) -> Path:
        return Path(
            os.getenv(
                "REPORTS_DIR",
                self.repo_root / "src" / "reports",
            )
        )

    @property
    def templates_dir(self) -> Path:
        return self.repo_root / "webapp" / "web" / "templates"

    @property
    def static_dir(self) -> Path:
        return self.repo_root / "webapp" / "web" / "static"


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "5"))

    return Settings(
        repo_root=repo_root,
        app_name=constants.APP_NAME,
        version=constants.APP_VERSION,
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        default_model_id=os.getenv("DEFAULT_MODEL_ID", constants.DEFAULT_MODEL_ID),
        normalization_mean=constants.NORMALIZATION_MEAN,
        normalization_std=constants.NORMALIZATION_STD,
    )


settings = load_settings()
