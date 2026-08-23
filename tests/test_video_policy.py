from pathlib import Path

import pytest

from ms_peso.service.config import PROJECT_ROOT
from ms_peso.service.video_policy import load_video_inference_policy


def test_default_video_policy_keeps_uncalibrated_controls_explicit() -> None:
    policy = load_video_inference_policy(
        PROJECT_ROOT / "configs" / "video_inference.yaml"
    )
    assert policy.top_k == 5
    assert policy.min_valid_frames == 3
    assert policy.max_frame_pixels == 8294400
    assert policy.max_frame_spread_kg is None
    assert policy.status == "design_unvalidated"


def test_video_policy_rejects_inconsistent_frame_counts(tmp_path: Path) -> None:
    policy_path = tmp_path / "invalid.yaml"
    policy_path.write_text(
        """
schema_version: 1
policy_id: invalid
policy_version: "1"
status: design_unvalidated
extraction: {max_duration_seconds: 10, sample_count: 3, max_frame_pixels: 1000}
selection: {min_valid_frames: 4, top_k: 5, min_temporal_gap_seconds: 0.2}
consensus: {max_frame_spread_kg: null}
limitations: [Ainda não validado.]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_valid_frames"):
        load_video_inference_policy(policy_path)
