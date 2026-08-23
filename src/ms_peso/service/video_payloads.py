from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ms_peso.service.payloads import build_model_payload
from ms_peso.service.uploads import StoredUpload
from ms_peso.service.video_inference import VideoPredictionResult


def _interval_payload(result) -> dict[str, object]:
    return {
        "lower_kg": round(result.interval_lower_kg, 4),
        "upper_kg": round(result.interval_upper_kg, 4),
        "radius_kg": round(result.interval_radius_kg, 4),
        "target_coverage": result.target_coverage,
        "method": "split_conformal_grouped_absolute_residual",
        "scope": "individual_extracted_frame_only",
        "lower_bound_clipped_at_zero": result.lower_bound_clipped_at_zero,
    }


def build_video_prediction_payload(
    result: VideoPredictionResult,
    upload: StoredUpload,
    *,
    correlation_id: str | None,
) -> dict[str, object]:
    first_backend_result = result.frames[0].backend_result
    descriptor = first_backend_result.descriptor
    warnings = tuple(
        dict.fromkeys((*result.policy.limitations, *descriptor.limitations))
    )
    rejection = (
        {"code": result.rejection_code, "detail": result.rejection_detail}
        if result.rejection_code
        else None
    )
    frame_payloads = []
    for item in result.frames:
        prediction = item.backend_result.prediction
        if prediction is None:
            raise RuntimeError("Resultado de quadro sem previsão.")
        frame_payloads.append(
            {
                "frame_index": item.assessed.frame.frame_index,
                "timestamp_seconds": round(
                    item.assessed.frame.timestamp_seconds, 4
                ),
                "technical_quality_score": item.assessed.technical_score,
                "estimated_weight_kg": round(prediction.estimated_weight_kg, 4),
                "prediction_interval": _interval_payload(prediction),
                "quality": item.backend_result.quality.to_dict(),
            }
        )

    return {
        "schema_version": 1,
        "prediction_id": str(uuid4()),
        "correlation_id": correlation_id,
        "created_at": datetime.now(UTC).isoformat(),
        "prediction_status": result.prediction_status,
        "rejection": rejection,
        "authorization_status": "blocked_pending_mandatory_reviews",
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
            "max_frame_spread_kg": result.policy.max_frame_spread_kg,
        },
        "input": {
            "filename": upload.original_filename,
            "content_type": upload.content_type,
            "size_bytes": upload.size_bytes,
            "duration_seconds": result.metadata.duration_seconds,
            "fps": result.metadata.fps,
            "frame_count": result.metadata.frame_count,
            "width": result.metadata.width,
            "height": result.metadata.height,
            "expected_view": descriptor.input_view,
        },
        "frames": frame_payloads,
        "video_policy": {
            "id": result.policy.policy_id,
            "version": result.policy.policy_version,
            "status": result.policy.status,
        },
        "model": build_model_payload(first_backend_result),
        "warnings": list(warnings),
    }
