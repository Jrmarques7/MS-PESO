import pytest

from ms_peso.conformal import calibrate_grouped_absolute_residuals


def test_uses_maximum_residual_per_animal() -> None:
    result = calibrate_grouped_absolute_residuals(
        targets=[100.0, 100.0, 100.0, 100.0, 100.0],
        predictions=[99.0, 110.0, 102.0, 103.0, 104.0],
        group_ids=["animal_a", "animal_a", "animal_b", "animal_c", "animal_d"],
        target_coverage=0.80,
    )

    assert result.number_of_groups == 4
    assert result.quantile_rank == 4
    assert result.radius_kg == pytest.approx(10.0)
    assert result.empirical_group_coverage == pytest.approx(1.0)


def test_rejects_coverage_unsupported_by_number_of_animals() -> None:
    with pytest.raises(ValueError, match="Animais de calibração insuficientes"):
        calibrate_grouped_absolute_residuals(
            targets=[100.0] * 8,
            predictions=[101.0] * 8,
            group_ids=[f"animal_{index}" for index in range(8)],
            target_coverage=0.90,
        )
