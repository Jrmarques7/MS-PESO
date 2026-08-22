import math

import pytest

from ms_peso.metrics import regression_metrics


def test_perfect_predictions():
    metrics = regression_metrics([300, 400, 500], [300, 400, 500])
    assert metrics["mae_kg"] == 0
    assert metrics["rmse_kg"] == 0
    assert metrics["mape_pct"] == 0
    assert metrics["r2"] == 1
    assert metrics["within_5kg_pct"] == 100


def test_known_errors():
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae_kg"] == 15
    assert metrics["rmse_kg"] == pytest.approx(math.sqrt(250))
    assert metrics["bias_kg"] == -5
    assert metrics["mape_pct"] == pytest.approx(10)
    assert metrics["within_20kg_pct"] == 100


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        regression_metrics([100], [100, 200])
