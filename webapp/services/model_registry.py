"""Checkpoint loading and model registry."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from webapp.core.constants import CIFAR10_CLASSES, MODEL_CHECKPOINT_FILENAMES
from webapp.core.config import Settings
from webapp.models.cnn import create_model
from webapp.schemas.prediction import ModelId


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[ModelId, nn.Module] = {}
        self._checkpoint_paths: dict[ModelId, Path] = {}
        self.device = torch.device("cpu")

    @property
    def loaded_model_ids(self) -> list[ModelId]:
        return list(self._models.keys())

    def load_all(self) -> None:
        missing_paths: list[Path] = []
        for raw_model_id, checkpoint_name in MODEL_CHECKPOINT_FILENAMES.items():
            model_id = ModelId(raw_model_id)
            checkpoint_path = self.settings.checkpoints_dir / checkpoint_name
            if not checkpoint_path.exists():
                missing_paths.append(checkpoint_path)
                continue

            model = create_model(model_id).to(self.device)
            checkpoint_obj = torch.load(checkpoint_path, map_location=self.device)
            state_dict = self._extract_state_dict(checkpoint_obj)
            if not isinstance(state_dict, dict):
                raise RuntimeError(f"Invalid checkpoint format for {checkpoint_path}.")

            normalized_state = {
                str(key).removeprefix("module."): value
                for key, value in state_dict.items()
            }
            model.load_state_dict(normalized_state)
            model.eval()

            self._models[model_id] = model
            self._checkpoint_paths[model_id] = checkpoint_path

        if missing_paths:
            expected = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(f"Missing checkpoint files at startup: {expected}")

    def get_model(self, model_id: ModelId) -> nn.Module:
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"Model {model_id} is not loaded.")
        return model

    @staticmethod
    def _extract_state_dict(checkpoint_obj: object) -> dict[str, torch.Tensor] | None:
        if isinstance(checkpoint_obj, dict):
            for key in ("model_state_dict", "state_dict"):
                value = checkpoint_obj.get(key)
                if isinstance(value, dict):
                    return value
            # Some checkpoints are raw state_dicts without wrapper metadata.
            if checkpoint_obj and all(isinstance(k, str) for k in checkpoint_obj):
                return checkpoint_obj  # type: ignore[return-value]
            if not checkpoint_obj:
                return checkpoint_obj  # type: ignore[return-value]
        return None

    def list_model_metadata(self) -> list[dict[str, str | int | ModelId]]:
        return [
            {
                "id": model_id,
                "checkpoint": str(self._checkpoint_paths[model_id].name),
                "classes_count": len(CIFAR10_CLASSES),
            }
            for model_id in self.loaded_model_ids
        ]
