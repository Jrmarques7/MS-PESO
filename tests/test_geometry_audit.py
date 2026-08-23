import pytest

from ms_peso.audit_point_cloud_geometry import (
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
