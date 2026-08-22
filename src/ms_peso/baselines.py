from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from ms_peso.metrics import regression_metrics


@dataclass(frozen=True)
class MeanBaselineEvaluation:
    """Resultado do preditor constante ajustado somente no conjunto de treino."""

    training_mean_kg: float
    targets: list[float]
    predictions: list[float]
    metrics: dict[str, float]


def evaluate_mean_baseline(
    training_targets: list[float],
    test_targets: list[float],
) -> MeanBaselineEvaluation:
    """Avalia exclusivamente a média do treino como predição constante."""
    if not training_targets:
        raise ValueError("O baseline da média requer pesos de treino.")
    if not test_targets:
        raise ValueError("O baseline da média requer pesos de teste.")

    training_mean = fmean(float(value) for value in training_targets)
    targets = [float(value) for value in test_targets]
    predictions = [training_mean] * len(targets)
    return MeanBaselineEvaluation(
        training_mean_kg=training_mean,
        targets=targets,
        predictions=predictions,
        metrics=regression_metrics(targets, predictions),
    )
