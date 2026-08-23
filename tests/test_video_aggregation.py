from ms_peso.service.video_aggregation import aggregate_frame_weights


def test_uses_median_and_exposes_uncalibrated_consensus() -> None:
    result = aggregate_frame_weights(
        [398.0, 400.0, 402.0, 405.0, 900.0],
        max_frame_spread_kg=None,
    )
    assert result.estimated_weight_kg == 402.0
    assert result.frame_spread_kg == 502.0
    assert result.median_absolute_deviation_kg == 3.0
    assert result.consensus_status == "threshold_not_calibrated"


def test_rejects_divergence_only_when_policy_has_calibrated_threshold() -> None:
    result = aggregate_frame_weights(
        [390.0, 400.0, 410.0],
        max_frame_spread_kg=15.0,
    )
    assert result.prediction_status == "rejected"
    assert result.estimated_weight_kg is None
    assert result.rejection_code == "inconsistent_frame_predictions"
