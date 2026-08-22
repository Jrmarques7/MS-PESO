import csv

import pytest

from ms_peso.error_analysis import (
    Prediction,
    load_predictions,
    summarize,
    summarize_by_weight_band,
)


def test_summarize_reports_bias_and_direction_counts() -> None:
    predictions = [Prediction("a", 100.0, 110.0), Prediction("b", 200.0, 180.0)]

    result = summarize(predictions)

    assert result["mae_kg"] == 15.0
    assert result["bias_kg"] == -5.0
    assert result["overestimated_animals"] == 1
    assert result["underestimated_animals"] == 1
    assert result["mape_pct"] == pytest.approx(10.0)


def test_weight_bands_do_not_overlap() -> None:
    predictions = [
        Prediction("a", 349.0, 349.0),
        Prediction("b", 350.0, 350.0),
        Prediction("c", 400.0, 400.0),
        Prediction("d", 500.0, 500.0),
    ]

    result = summarize_by_weight_band(predictions)

    assert sum(int(row["animals"]) for row in result) == 4


def test_load_predictions_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file,
            fieldnames=["animal_id", "weight_kg", "predicted_weight_kg"],
        ).writeheader()

    with pytest.raises(ValueError, match="vazio"):
        load_predictions(path)
