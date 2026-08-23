import pytest

from ms_peso.audit_point_cloud_geometry import (
    align_reference_validation,
    fit_height_regression,
    pearson_correlation,
    predict_height_regression,
)


def test_fits_height_regression() -> None:
    intercept, slope = fit_height_regression([1.0, 2.0, 3.0], [100, 200, 300])

    predictions = predict_height_regression(
        [1.5, 2.5], intercept=intercept, slope=slope
    )

    assert predictions == pytest.approx([150, 250])


def test_calculates_pearson_correlation() -> None:
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)


def test_rejects_constant_correlation_input() -> None:
    with pytest.raises(ValueError, match="variação"):
        pearson_correlation([1, 1, 1], [2, 3, 4])


def test_rejects_constant_height_regression() -> None:
    with pytest.raises(ValueError, match="variação"):
        fit_height_regression([1, 1, 1], [100, 200, 300])


def test_aligns_reference_and_geometry_by_event() -> None:
    geometry_rows = [
        {
            "animal_id": "cow_2",
            "event_id": "event_2",
            "weight_kg": "420",
            "split": "val",
        },
        {
            "animal_id": "cow_1",
            "event_id": "event_1",
            "weight_kg": "400",
            "split": "val",
        },
    ]
    reference_rows = list(reversed(geometry_rows))

    rows, heights = align_reference_validation(
        reference_rows, geometry_rows, [1.2, 1.0]
    )

    assert [row["animal_id"] for row in rows] == ["cow_1", "cow_2"]
    assert heights == [1.0, 1.2]


def test_rejects_missing_reference_event() -> None:
    geometry_rows = [
        {
            "animal_id": "cow_1",
            "event_id": "event_1",
            "weight_kg": "400",
            "split": "val",
        }
    ]

    with pytest.raises(ValueError, match="ausentes"):
        align_reference_validation([], geometry_rows, [1.0])
