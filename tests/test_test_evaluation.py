import math

import pytest

from ms_peso.test_evaluation import (
    cluster_bootstrap_metrics,
    evaluate_symmetric_interval_coverage,
    grouped_regression_metrics,
)


def test_grouped_metrics_give_equal_weight_to_each_animal() -> None:
    metrics = grouped_regression_metrics(
        targets=[100.0, 100.0, 100.0],
        predictions=[100.0, 110.0, 120.0],
        group_ids=["animal_a", "animal_a", "animal_b"],
    )

    assert metrics["mae_kg"] == pytest.approx(12.5)
    assert metrics["rmse_kg"] == pytest.approx(15.0)
    assert metrics["mape_pct"] == pytest.approx(12.5)
    assert metrics["bias_kg"] == pytest.approx(12.5)


def test_cluster_bootstrap_is_reproducible() -> None:
    arguments = {
        "targets": [100.0, 120.0, 140.0, 160.0],
        "predictions": [105.0, 110.0, 150.0, 155.0],
        "group_ids": ["a", "b", "c", "d"],
        "iterations": 100,
        "seed": 42,
    }

    first = cluster_bootstrap_metrics(**arguments)
    second = cluster_bootstrap_metrics(**arguments)

    assert first == second
    assert first["mae_kg"]["point"] == pytest.approx(7.5)
    assert all(
        math.isfinite(value)
        for metric in first.values()
        for value in metric.values()
    )


def test_interval_coverage_uses_worst_image_per_animal() -> None:
    coverage = evaluate_symmetric_interval_coverage(
        targets=[100.0, 100.0, 100.0],
        predictions=[101.0, 105.0, 102.0],
        group_ids=["animal_a", "animal_a", "animal_b"],
        radius_kg=4.0,
    )

    assert coverage.covered_images == 2
    assert coverage.image_coverage == pytest.approx(2 / 3)
    assert coverage.covered_groups == 1
    assert coverage.group_coverage == pytest.approx(0.5)
    assert 0 <= coverage.group_coverage_lower_95 < 0.5
    assert 0.5 < coverage.group_coverage_upper_95 <= 1
