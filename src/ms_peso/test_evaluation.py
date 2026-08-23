from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

METRIC_NAMES = ("mae_kg", "rmse_kg", "mape_pct", "bias_kg")


@dataclass(frozen=True)
class GroupSummary:
    mean_absolute_error: float
    mean_squared_error: float
    mean_percentage_error: float
    mean_error: float


@dataclass(frozen=True)
class IntervalCoverage:
    radius_kg: float
    number_of_images: int
    covered_images: int
    image_coverage: float
    number_of_groups: int
    covered_groups: int
    group_coverage: float
    group_coverage_lower_95: float
    group_coverage_upper_95: float


def _validate_inputs(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
) -> None:
    if not (len(targets) == len(predictions) == len(group_ids)) or not targets:
        raise ValueError(
            "targets, predictions e group_ids devem ter o mesmo tamanho não vazio."
        )
    if any(not group_id.strip() for group_id in group_ids):
        raise ValueError("group_ids não pode conter valores vazios.")
    if any(not math.isfinite(value) for value in (*targets, *predictions)):
        raise ValueError("Alvos e predições precisam ser finitos.")
    if any(target <= 0 for target in targets):
        raise ValueError("Alvos precisam ser positivos para calcular MAPE.")


def _group_summaries(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
) -> dict[str, GroupSummary]:
    _validate_inputs(targets, predictions, group_ids)
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for target, prediction, group_id in zip(
        targets, predictions, group_ids, strict=True
    ):
        values[group_id].append((target, prediction))

    summaries = {}
    for group_id, pairs in values.items():
        errors = [prediction - target for target, prediction in pairs]
        summaries[group_id] = GroupSummary(
            mean_absolute_error=sum(abs(error) for error in errors) / len(errors),
            mean_squared_error=sum(error**2 for error in errors) / len(errors),
            mean_percentage_error=(
                100
                * sum(abs(prediction - target) / target for target, prediction in pairs)
                / len(pairs)
            ),
            mean_error=sum(errors) / len(errors),
        )
    return summaries


def _metrics_from_summaries(items: Sequence[GroupSummary]) -> dict[str, float]:
    if not items:
        raise ValueError("Ao menos um grupo é necessário para calcular métricas.")
    return {
        "mae_kg": sum(item.mean_absolute_error for item in items) / len(items),
        "rmse_kg": math.sqrt(
            sum(item.mean_squared_error for item in items) / len(items)
        ),
        "mape_pct": sum(item.mean_percentage_error for item in items) / len(items),
        "bias_kg": sum(item.mean_error for item in items) / len(items),
    }


def grouped_regression_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
) -> dict[str, float]:
    """Calcula métricas dando o mesmo peso estatístico a cada animal."""
    summaries = _group_summaries(targets, predictions, group_ids)
    return _metrics_from_summaries(list(summaries.values()))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def cluster_bootstrap_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Estima IC95% reamostrando animais inteiros com reposição."""
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations deve ser um inteiro.")
    if iterations < 100:
        raise ValueError("iterations deve ser pelo menos 100.")
    summaries = _group_summaries(targets, predictions, group_ids)
    items = list(summaries.values())
    point = _metrics_from_summaries(items)
    distributions = {name: [] for name in METRIC_NAMES}
    rng = random.Random(seed)
    for _ in range(iterations):
        sample = [items[rng.randrange(len(items))] for _ in items]
        sample_metrics = _metrics_from_summaries(sample)
        for name in METRIC_NAMES:
            distributions[name].append(sample_metrics[name])
    return {
        name: {
            "point": point[name],
            "lower_95": _percentile(sorted(distributions[name]), 0.025),
            "upper_95": _percentile(sorted(distributions[name]), 0.975),
        }
        for name in METRIC_NAMES
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Contagens inválidas para intervalo de Wilson.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_symmetric_interval_coverage(
    targets: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[str],
    *,
    radius_kg: float,
) -> IntervalCoverage:
    """Mede cobertura por imagem e pelo pior resíduo de cada animal."""
    _validate_inputs(targets, predictions, group_ids)
    if not math.isfinite(radius_kg) or radius_kg < 0:
        raise ValueError("radius_kg deve ser finito e não negativo.")
    covered_images = 0
    maximum_residual_by_group: dict[str, float] = {}
    for target, prediction, group_id in zip(
        targets, predictions, group_ids, strict=True
    ):
        residual = abs(prediction - target)
        covered_images += residual <= radius_kg
        maximum_residual_by_group[group_id] = max(
            maximum_residual_by_group.get(group_id, 0.0), residual
        )
    covered_groups = sum(
        residual <= radius_kg for residual in maximum_residual_by_group.values()
    )
    number_of_groups = len(maximum_residual_by_group)
    lower, upper = _wilson_interval(covered_groups, number_of_groups)
    return IntervalCoverage(
        radius_kg=radius_kg,
        number_of_images=len(targets),
        covered_images=covered_images,
        image_coverage=covered_images / len(targets),
        number_of_groups=number_of_groups,
        covered_groups=covered_groups,
        group_coverage=covered_groups / number_of_groups,
        group_coverage_lower_95=lower,
        group_coverage_upper_95=upper,
    )
