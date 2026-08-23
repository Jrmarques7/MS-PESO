from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConformalCalibration:
    """Raio conformal calculado com grupos independentes."""

    target_coverage: float
    alpha: float
    number_of_groups: int
    quantile_rank: int
    radius_kg: float
    empirical_group_coverage: float


def validate_target_coverage(target_coverage: float) -> float:
    """Valida exclusivamente uma cobertura conformal solicitada."""
    if isinstance(target_coverage, bool) or not math.isfinite(target_coverage):
        raise ValueError("target_coverage deve ser um número finito entre 0 e 1.")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage deve estar estritamente entre 0 e 1.")
    return target_coverage


def conformal_quantile_rank(
    number_of_groups: int, *, target_coverage: float
) -> int:
    """Valida a cobertura e retorna a ordem conservadora do quantil finito."""
    if (
        isinstance(number_of_groups, bool)
        or not isinstance(number_of_groups, int)
        or number_of_groups <= 0
    ):
        raise ValueError("number_of_groups deve ser um inteiro positivo.")
    validate_target_coverage(target_coverage)
    quantile_rank = math.ceil((number_of_groups + 1) * target_coverage)
    if quantile_rank > number_of_groups:
        raise ValueError(
            "Animais de calibração insuficientes para a cobertura solicitada: "
            f"n={number_of_groups}, cobertura={target_coverage:.4f}."
        )
    return quantile_rank


def calibrate_grouped_absolute_residuals(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
    *,
    target_coverage: float,
) -> ConformalCalibration:
    """Calcula split conformal usando o maior erro absoluto de cada grupo."""
    if not (len(targets) == len(predictions) == len(group_ids)):
        raise ValueError("targets, predictions e group_ids devem ter o mesmo tamanho.")
    if not targets:
        raise ValueError("A calibração exige ao menos uma predição.")
    scores_by_group: dict[str, float] = {}
    for target, prediction, group_id in zip(
        targets, predictions, group_ids, strict=True
    ):
        if not group_id.strip():
            raise ValueError("group_ids não pode conter identificadores vazios.")
        if not math.isfinite(target) or not math.isfinite(prediction):
            raise ValueError("Alvos e predições precisam ser finitos.")
        residual = abs(prediction - target)
        scores_by_group[group_id] = max(scores_by_group.get(group_id, 0.0), residual)

    ordered_scores = sorted(scores_by_group.values())
    number_of_groups = len(ordered_scores)
    quantile_rank = conformal_quantile_rank(
        number_of_groups, target_coverage=target_coverage
    )
    radius_kg = ordered_scores[quantile_rank - 1]
    empirical_coverage = (
        sum(score <= radius_kg for score in ordered_scores) / number_of_groups
    )
    return ConformalCalibration(
        target_coverage=target_coverage,
        alpha=1.0 - target_coverage,
        number_of_groups=number_of_groups,
        quantile_rank=quantile_rank,
        radius_kg=radius_kg,
        empirical_group_coverage=empirical_coverage,
    )
