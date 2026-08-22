import pytest

from ms_peso.sampling import inverse_frequency_weights, weight_band_index


def test_weight_band_boundaries_are_left_inclusive() -> None:
    boundaries = [350, 400, 450]
    assert weight_band_index(349.9, boundaries) == 0
    assert weight_band_index(350, boundaries) == 1
    assert weight_band_index(450, boundaries) == 3


def test_rare_band_receives_larger_sampling_weight() -> None:
    weights = [300, 360, 370, 380, 410, 420]
    result = inverse_frequency_weights(weights, boundaries=[350, 400, 450], power=0.5)
    assert result[0] > result[1]
    assert result[4] > result[1]


@pytest.mark.parametrize("power", [0, 1.1])
def test_rejects_invalid_power(power: float) -> None:
    with pytest.raises(ValueError, match="power"):
        inverse_frequency_weights([300], boundaries=[350], power=power)
