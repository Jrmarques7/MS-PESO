import csv

import pytest

from ms_peso.artifacts import save_predictions


def test_saves_generic_predictions_without_torch(tmp_path):
    output = tmp_path / "predictions.csv"
    rows = [
        {
            "image_path": "cow.png",
            "animal_id": "cow_001",
            "event_id": "event_001",
        }
    ]

    save_predictions(
        output,
        rows,
        targets=[410.0],
        predictions=[400.0],
        indices=[0],
    )

    with output.open(encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    assert saved[0]["weight_kg"] == "410.0000"
    assert saved[0]["predicted_weight_kg"] == "400.0000"
    assert saved[0]["error_kg"] == "-10.0000"


def test_rejects_prediction_vectors_with_different_sizes(tmp_path):
    with pytest.raises(ValueError, match="mesmo tamanho"):
        save_predictions(
            tmp_path / "predictions.csv",
            [],
            targets=[400.0],
            predictions=[],
            indices=[0],
        )
