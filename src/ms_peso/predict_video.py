from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ms_peso.artifacts import save_json
from ms_peso.image_quality import load_image_quality_policy
from ms_peso.model_package import load_model_descriptor
from ms_peso.research_video_inference import (
    B2ResearchVideoPredictor,
    ResearchVideoPrediction,
)
from ms_peso.service.video_frames import VideoValidationError
from ms_peso.service.video_policy import load_video_inference_policy
from ms_peso.service.video_selection import FrameSelectionError


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima peso experimental a partir de vídeo lateral curto."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", default="models/b2_cowdb.yaml", type=Path)
    parser.add_argument(
        "--quality-policy", default="configs/image_quality.yaml", type=Path
    )
    parser.add_argument(
        "--video-policy", default="configs/video_inference.yaml", type=Path
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-frames-dir", type=Path)
    return parser.parse_args(args)


def build_payload(
    result: ResearchVideoPrediction, *, video_path: Path
) -> dict[str, object]:
    descriptor = result.descriptor
    rejection = (
        {"code": result.rejection_code, "detail": result.rejection_detail}
        if result.rejection_code
        else None
    )
    return {
        "schema_version": 1,
        "prediction_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "prediction_status": result.prediction_status,
        "rejection": rejection,
        "authorization_status": "research_only_blocked_for_commercial_use",
        "estimated_weight_kg": (
            round(result.estimated_weight_kg, 4)
            if result.estimated_weight_kg is not None
            else None
        ),
        "prediction_interval": None,
        "interval_status": "pending_video_calibration",
        "aggregation": {
            "method": "median_selected_frames",
            "sampled_frame_count": result.sampled_frame_count,
            "technically_eligible_frame_count": result.eligible_frame_count,
            "selected_frame_count": len(result.frames),
            "frame_spread_kg": round(result.frame_spread_kg, 4),
            "median_absolute_deviation_kg": round(
                result.median_absolute_deviation_kg, 4
            ),
            "consensus_status": result.consensus_status,
            "max_frame_spread_kg": result.video_policy.max_frame_spread_kg,
        },
        "input": {
            "video_path": str(video_path.resolve()),
            "duration_seconds": result.metadata.duration_seconds,
            "fps": result.metadata.fps,
            "frame_count": result.metadata.frame_count,
            "width": result.metadata.width,
            "height": result.metadata.height,
            "expected_view": descriptor.input_view,
        },
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp_seconds": round(frame.timestamp_seconds, 4),
                "technical_quality_score": frame.technical_score,
                "estimated_weight_kg": round(frame.estimated_weight_kg, 4),
                "quality": frame.quality.to_dict(),
                "exported_path": (
                    str(frame.exported_path) if frame.exported_path else None
                ),
            }
            for frame in result.frames
        ],
        "video_policy": {
            "id": result.video_policy.policy_id,
            "version": result.video_policy.policy_version,
            "status": result.video_policy.status,
        },
        "model": {
            "id": descriptor.model_id,
            "version": descriptor.model_version,
            "architecture": descriptor.architecture,
            "checkpoint_sha256": descriptor.checkpoint_sha256,
            "status": descriptor.status,
            "production_ready": descriptor.production_ready,
            "commercial_use_allowed": descriptor.commercial_use_allowed,
            "commercial_blockers": list(descriptor.commercial_blockers),
            "training_dataset": descriptor.dataset,
            "training_breed": descriptor.breed,
            "device": result.device,
        },
        "warnings": list(
            dict.fromkeys(
                (*descriptor.limitations, *result.video_policy.limitations)
            )
        ),
    }


def main() -> None:
    args = parse_args()
    try:
        predictor = B2ResearchVideoPredictor(
            descriptor=load_model_descriptor(args.model),
            quality_policy=load_image_quality_policy(args.quality_policy),
            video_policy=load_video_inference_policy(args.video_policy),
            device=args.device,
        )
        result = predictor.predict(
            args.video,
            selected_frames_directory=args.selected_frames_dir,
        )
        payload = build_payload(result, video_path=args.video)
    except (FrameSelectionError, VideoValidationError, ValueError) as exc:
        payload = {
            "schema_version": 1,
            "prediction_status": "rejected",
            "authorization_status": "research_only_blocked_for_commercial_use",
            "estimated_weight_kg": None,
            "error": {"code": type(exc).__name__, "detail": str(exc)},
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["prediction_status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
