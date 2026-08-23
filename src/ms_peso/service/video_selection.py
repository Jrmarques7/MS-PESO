from __future__ import annotations

import math
from dataclasses import dataclass

from ms_peso.image_quality import ImageQualityReport
from ms_peso.service.video_frames import ExtractedFrame
from ms_peso.service.video_policy import VideoInferencePolicy


class FrameSelectionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class AssessedFrame:
    frame: ExtractedFrame
    quality: ImageQualityReport
    technical_score: float


@dataclass(frozen=True)
class FrameSelection:
    assessed_count: int
    eligible_count: int
    selected: tuple[AssessedFrame, ...]


def _numeric_check(report: ImageQualityReport, code: str) -> float:
    for check in report.checks:
        if check.code == code and isinstance(check.value, (int, float)):
            return float(check.value)
    return 0.0


def technical_quality_score(report: ImageQualityReport) -> float:
    """Rank accepted frames using only measurements already in the gate."""

    if not report.accepted:
        return float("-inf")
    sharpness = max(0.0, _numeric_check(report, "sharpness"))
    mean_luma = _numeric_check(report, "mean_luma")
    dark_fraction = max(0.0, _numeric_check(report, "dark_clipping"))
    bright_fraction = max(0.0, _numeric_check(report, "bright_clipping"))
    exposure_penalty = abs(mean_luma - 127.5) / 127.5
    clipping_penalty = 2.0 * (dark_fraction + bright_fraction)
    return round(math.log1p(sharpness) - exposure_penalty - clipping_penalty, 6)


def select_technical_frames(
    assessed_frames: tuple[AssessedFrame, ...],
    policy: VideoInferencePolicy,
) -> FrameSelection:
    eligible = [item for item in assessed_frames if item.quality.accepted]
    ranked = sorted(
        eligible,
        key=lambda item: (-item.technical_score, item.frame.timestamp_seconds),
    )
    selected: list[AssessedFrame] = []
    for candidate in ranked:
        timestamp = candidate.frame.timestamp_seconds
        if all(
            abs(timestamp - item.frame.timestamp_seconds)
            >= policy.min_temporal_gap_seconds
            for item in selected
        ):
            selected.append(candidate)
        if len(selected) == policy.top_k:
            break

    if len(selected) < policy.min_valid_frames:
        code = (
            "insufficient_quality_frames"
            if len(eligible) < policy.min_valid_frames
            else "insufficient_temporal_diversity"
        )
        raise FrameSelectionError(
            code,
            (
                "O vídeo não forneceu quadros técnicos e temporalmente diversos "
                f"suficientes: {len(selected)} de pelo menos "
                f"{policy.min_valid_frames}."
            ),
        )

    selected.sort(key=lambda item: item.frame.timestamp_seconds)
    return FrameSelection(
        assessed_count=len(assessed_frames),
        eligible_count=len(eligible),
        selected=tuple(selected),
    )
