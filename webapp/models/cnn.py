"""CNN model definitions extracted from the training notebook."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from webapp.schemas.prediction import ModelId


class BaselineCNN(nn.Module):
    """Notebook baseline architecture used for saved checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNNV2(BaselineCNN):
    """V2 checkpoint currently shares the same architecture for inference."""


def create_model(model_id: ModelId) -> nn.Module:
    if model_id == ModelId.baseline:
        return BaselineCNN()
    if model_id == ModelId.cnnv2:
        return CNNV2()
    raise ValueError(f"Unsupported model id: {model_id}")
