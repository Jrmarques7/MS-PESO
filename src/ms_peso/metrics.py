from __future__ import annotations

import math
from collections.abc import Sequence


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets e predictions devem ter o mesmo tamanho não vazio")

    target_values = [float(value) for value in targets]
    prediction_values = [float(value) for value in predictions]
    errors = [
        prediction - target
        for target, prediction in zip(target_values, prediction_values, strict=True)
    ]
    absolute_errors = [abs(error) for error in errors]
    count = len(errors)
    mean_target = sum(target_values) / count
    total_variance = sum((target - mean_target) ** 2 for target in target_values)
    residual_variance = sum(error**2 for error in errors)

    metrics = {
        "mae_kg": sum(absolute_errors) / count,
        "rmse_kg": math.sqrt(residual_variance / count),
        "bias_kg": sum(errors) / count,
        "r2": (
            1.0 - residual_variance / total_variance
            if total_variance > 0
            else float("nan")
        ),
        "within_5kg_pct": 100 * sum(error <= 5 for error in absolute_errors) / count,
        "within_10kg_pct": 100 * sum(error <= 10 for error in absolute_errors) / count,
        "within_20kg_pct": 100 * sum(error <= 20 for error in absolute_errors) / count,
    }
    positive_pairs = [
        (target, error)
        for target, error in zip(target_values, absolute_errors, strict=True)
        if target > 0
    ]
    metrics["mape_pct"] = (
        100
        * sum(error / target for target, error in positive_pairs)
        / len(positive_pairs)
        if positive_pairs
        else float("nan")
    )
    return metrics
