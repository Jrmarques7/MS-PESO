import pytest

from ms_peso.compare_models import AnimalPrediction, compare, percentile


def _predictions(values: list[float]) -> dict[str, AnimalPrediction]:
    return {
        str(index): AnimalPrediction(str(index), 100.0, prediction)
        for index, prediction in enumerate(values)
    }


def test_paired_bootstrap_detects_uniform_improvement() -> None:
    reference = _predictions([120, 120, 120, 120])
    candidate = _predictions([110, 110, 110, 110])

    result = compare(reference, candidate, iterations=200, seed=7)

    mae = result["metrics"]["mae_kg"]
    assert mae["delta_candidate_minus_reference"]["point"] == -10
    assert mae["probability_candidate_lower"] == 1


def test_rejects_different_animals() -> None:
    with pytest.raises(ValueError, match="mesmos animal_id"):
        compare(_predictions([110]), {}, iterations=100)


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0], 0.25) == 2.5
