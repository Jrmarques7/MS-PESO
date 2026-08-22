import pytest

from ms_peso.baselines import evaluate_mean_baseline


def test_mean_baseline_uses_training_mean_for_every_test_sample():
    result = evaluate_mean_baseline([300, 400, 500], [350, 450])

    assert result.training_mean_kg == 400
    assert result.predictions == [400, 400]
    assert result.targets == [350, 450]
    assert result.metrics["mae_kg"] == 50
    assert result.metrics["bias_kg"] == 0


@pytest.mark.parametrize(
    ("training_targets", "test_targets", "message"),
    [
        ([], [300], "pesos de treino"),
        ([300], [], "pesos de teste"),
    ],
)
def test_mean_baseline_rejects_empty_partition(training_targets, test_targets, message):
    with pytest.raises(ValueError, match=message):
        evaluate_mean_baseline(training_targets, test_targets)
