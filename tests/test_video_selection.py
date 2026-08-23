from pathlib import Path

from ms_peso.image_quality import ImageQualityReport, QualityCheck
from ms_peso.service.video_frames import ExtractedFrame
from ms_peso.service.video_policy import VideoInferencePolicy
from ms_peso.service.video_selection import AssessedFrame, select_technical_frames


def _policy() -> VideoInferencePolicy:
    return VideoInferencePolicy(
        policy_id="test",
        policy_version="1",
        status="design_unvalidated",
        max_duration_seconds=10,
        sample_count=5,
        max_frame_pixels=1920 * 1080,
        min_valid_frames=3,
        top_k=3,
        min_temporal_gap_seconds=0.3,
        max_frame_spread_kg=None,
        limitations=("Teste.",),
    )


def _quality(accepted: bool) -> ImageQualityReport:
    return ImageQualityReport(
        policy_id="test",
        policy_version="1",
        width=96,
        height=64,
        checks=(
            QualityCheck(
                code="sharpness",
                passed=accepted,
                value=100.0,
                requirement=">= 10",
                rejection_message="Borrado.",
            ),
        ),
        limitations=("Não detecta pose.",),
    )


def _assessed(index: int, timestamp: float, score: float, accepted=True):
    return AssessedFrame(
        frame=ExtractedFrame(
            sample_index=index,
            frame_index=index * 10,
            timestamp_seconds=timestamp,
            path=Path(f"frame-{index}.png"),
        ),
        quality=_quality(accepted),
        technical_score=score,
    )


def test_selects_best_accepted_frames_with_temporal_diversity() -> None:
    assessed = (
        _assessed(0, 0.0, 8.0),
        _assessed(1, 0.1, 10.0),
        _assessed(2, 0.5, 7.0),
        _assessed(3, 1.0, 9.0),
        _assessed(4, 1.5, 20.0, accepted=False),
    )
    result = select_technical_frames(assessed, _policy())

    assert result.eligible_count == 4
    assert [item.frame.sample_index for item in result.selected] == [1, 2, 3]
