from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ms_peso.metrics import regression_metrics


@dataclass(frozen=True)
class EvaluationResult:
    targets: list[float]
    predictions: list[float]
    indices: list[int]

    @property
    def metrics(self) -> dict[str, float]:
        return regression_metrics(self.targets, self.predictions)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    target_mean: float,
    target_std: float,
) -> EvaluationResult:
    """Executa exclusivamente inferência avaliada sobre um DataLoader."""
    model.eval()
    targets: list[float] = []
    predictions: list[float] = []
    indices: list[int] = []
    with torch.no_grad():
        for images, normalized_targets, batch_indices in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1).cpu()
            batch_targets = normalized_targets * target_std + target_mean
            batch_predictions = outputs * target_std + target_mean
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.tolist())
            indices.extend(batch_indices.tolist())
    return EvaluationResult(targets, predictions, indices)
