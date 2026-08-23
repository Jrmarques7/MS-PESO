from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ms_peso.collection import (
    AuthorizationRecord,
    CollectionPolicy,
    audit_pilot_collection,
)
from ms_peso.collection_video import OUTPUT_COLUMNS, select_collection_frames
from ms_peso.image_quality import ImageQualityPolicy
from ms_peso.manifest import read_manifest
from ms_peso.service.video_policy import VideoInferencePolicy


def _collection_policy() -> CollectionPolicy:
    return CollectionPolicy(
        policy_id="pilot-test",
        policy_version="1",
        required_columns=OUTPUT_COLUMNS[:18],
        allowed_views=("left",),
        allowed_breeds=("nelore",),
        allowed_sexes=("male", "female"),
        min_weight_kg=100,
        max_weight_kg=1000,
        max_capture_weight_delta_minutes=30,
        required_quality="accepted",
        require_scale_marker=True,
        require_commercial_training_rights=True,
        near_duplicate_hamming_distance=4,
    )


def _quality_policy() -> ImageQualityPolicy:
    return ImageQualityPolicy(
        policy_id="quality-test",
        policy_version="1",
        analysis_max_dimension=128,
        min_width=96,
        min_height=64,
        min_aspect_ratio=1.4,
        max_aspect_ratio=1.6,
        min_mean_luma=20,
        max_mean_luma=230,
        max_dark_fraction=1,
        max_bright_fraction=1,
        min_sharpness=0,
        limitations=("Não confirma bovino ou pose.",),
    )


def _video_policy() -> VideoInferencePolicy:
    return VideoInferencePolicy(
        policy_id="video-test",
        policy_version="1",
        status="design_unvalidated",
        max_duration_seconds=5,
        sample_count=5,
        max_frame_pixels=96 * 64,
        min_valid_frames=3,
        top_k=3,
        min_temporal_gap_seconds=0.2,
        max_frame_spread_kg=None,
        limitations=("Teste.",),
    )


def _write_video(path: Path, *, checkerboard: bool) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(30):
        if checkerboard:
            yy, xx = np.indices((64, 96))
            gray = np.where((xx // 4 + yy // 4 + index) % 2, 220, 36).astype(
                np.uint8
            )
            frame = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            frame = np.full((64, 96, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _row(video_path: str, **updates: str) -> dict[str, str]:
    result = {
        "video_path": video_path,
        "identity_image_path": "",
        "animal_id": "nelore_001",
        "event_id": "event_001",
        "weight_kg": "600.0",
        "view": "left",
        "breed": "nelore",
        "sex": "male",
        "farm_id": "farm_001",
        "lot_id": "lot_001",
        "captured_at": "2026-08-23T12:07:51-03:00",
        "weighed_at": "2026-08-23T12:00:00-03:00",
        "camera_id": "camera_001",
        "scale_id": "scale_001",
        "duration_seconds": "3.0",
        "distance_m": "",
        "visible_animals": "1",
        "primary_full_body": "true",
        "primary_lateral": "true",
        "occlusion": "none",
        "lighting": "cloudy",
        "quality": "accepted",
        "scale_marker": "true",
        "authorization_id": "auth_001",
        "commercial_training_allowed": "true",
        "notes": "peso copiado da balança",
    }
    result.update(updates)
    return result


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(tmp_path: Path, rows: list[dict[str, str]]) -> tuple[dict[str, object], Path]:
    manifest = tmp_path / "pasture_video_manifest.csv"
    _write_manifest(manifest, rows)
    output = tmp_path / "interim/selection_v001"
    report = select_collection_frames(
        manifest_path=manifest,
        video_root=tmp_path / "raw",
        image_root=tmp_path,
        output_directory=output,
        collection_policy=_collection_policy(),
        quality_policy=_quality_policy(),
        video_policy=_video_policy(),
    )
    return report, output


def test_selects_one_best_frame_and_never_predicts_weight(tmp_path: Path) -> None:
    raw = tmp_path / "raw/farm_001/nelore_001/event_001"
    raw.mkdir(parents=True)
    _write_video(raw / "smooth.avi", checkerboard=False)
    _write_video(raw / "sharp.avi", checkerboard=True)

    report, output = _run(
        tmp_path,
        [
            _row("farm_001/nelore_001/event_001/smooth.avi"),
            _row("farm_001/nelore_001/event_001/sharp.avi"),
        ],
    )

    with (output / "pilot_manifest.csv").open(encoding="utf-8", newline="") as file:
        selected_rows = list(csv.DictReader(file))
    assert report["status"] == "completed"
    assert report["summary"]["candidate_videos"] == 2
    assert report["summary"]["selected_images"] == 1
    assert report["safety"]["weight_prediction_used"] is False
    assert len(selected_rows) == 1
    assert selected_rows[0]["weight_kg"] == "600.0"
    assert selected_rows[0]["quality"] == "review"
    assert selected_rows[0]["source_video_path"].endswith("sharp.avi")
    assert "predicted_weight_kg" not in selected_rows[0]
    selected_image = tmp_path / selected_rows[0]["image_path"]
    with Image.open(selected_image) as image:
        assert image.size == (96, 64)
    stored_report = json.loads((output / "selection_report.json").read_text())
    assert stored_report == report


def test_rejects_video_not_approved_by_human(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video(raw / "video.avi", checkerboard=True)

    report, output = _run(tmp_path, [_row("video.avi", quality="review")])

    assert report["status"] == "rejected"
    assert report["summary"]["selected_images"] == 0
    assert "após revisão humana" in report["rejections"][0]["detail"]
    assert not (output / "pilot_manifest.csv").exists()


def test_rejects_video_path_outside_root(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    outside = tmp_path / "outside.avi"
    _write_video(outside, checkerboard=True)

    report, _ = _run(tmp_path, [_row("../outside.avi")])

    assert report["status"] == "rejected"
    assert report["rejections"][0]["code"] == "invalid_source_row"
    assert "sair do --video-root" in report["rejections"][0]["detail"]


def test_refuses_to_overwrite_a_previous_selection(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video(raw / "video.avi", checkerboard=True)
    rows = [_row("video.avi")]

    _run(tmp_path, rows)
    with pytest.raises(FileExistsError, match="não será sobrescrita"):
        _run(tmp_path, rows)


def test_selected_frame_passes_collection_audit_only_after_human_approval(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video(raw / "video.avi", checkerboard=True)
    _, output = _run(tmp_path, [_row("video.avi")])
    manifest = output / "pilot_manifest.csv"
    rows = read_manifest(manifest)

    before_review = audit_pilot_collection(
        rows,
        {},
        _collection_policy(),
        manifest_path=manifest,
        image_root=tmp_path,
        check_images=True,
        quality_policy=_quality_policy(),
    )
    assert before_review.valid is False
    assert any("quality deve ser 'accepted'" in error for error in before_review.errors)

    rows[0]["quality"] = "accepted"
    authorization = AuthorizationRecord(
        authorization_id="auth_001",
        farm_id="farm_001",
        status="approved",
        effective_from=date(2026, 8, 1),
        effective_until=None,
        allows_model_training=True,
        allows_commercial_use=True,
        allows_data_sharing=False,
        document_reference="secure://auth_001",
    )
    after_review = audit_pilot_collection(
        rows,
        {"auth_001": authorization},
        _collection_policy(),
        manifest_path=manifest,
        image_root=tmp_path,
        check_images=True,
        quality_policy=_quality_policy(),
    )

    assert after_review.valid is True
    assert after_review.technical_quality_passed == 1
