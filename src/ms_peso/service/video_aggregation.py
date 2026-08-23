from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class VideoAggregate:
    prediction_status: str
    rejection_code: str | None
    rejection_detail: str | None
    estimated_weight_kg: float | None
    frame_spread_kg: float
    median_absolute_deviation_kg: float
    consensus_status: str


def aggregate_frame_weights(
    weights: list[float], *, max_frame_spread_kg: float | None
) -> VideoAggregate:
    if not weights:
        raise ValueError("A agregação de vídeo exige ao menos uma previsão.")

    aggregate = float(median(weights))
    spread = float(max(weights) - min(weights))
    absolute_deviations = [abs(value - aggregate) for value in weights]
    mad = float(median(absolute_deviations))

    if max_frame_spread_kg is None:
        return VideoAggregate(
            prediction_status="completed",
            rejection_code=None,
            rejection_detail=None,
            estimated_weight_kg=aggregate,
            frame_spread_kg=spread,
            median_absolute_deviation_kg=mad,
            consensus_status="threshold_not_calibrated",
        )
    if spread > max_frame_spread_kg:
        return VideoAggregate(
            prediction_status="rejected",
            rejection_code="inconsistent_frame_predictions",
            rejection_detail=(
                f"A divergência entre quadros foi {spread:.2f} kg; "
                f"o limite é {max_frame_spread_kg:.2f} kg."
            ),
            estimated_weight_kg=None,
            frame_spread_kg=spread,
            median_absolute_deviation_kg=mad,
            consensus_status="rejected",
        )
    return VideoAggregate(
        prediction_status="completed",
        rejection_code=None,
        rejection_detail=None,
        estimated_weight_kg=aggregate,
        frame_spread_kg=spread,
        median_absolute_deviation_kg=mad,
        consensus_status="passed",
    )
