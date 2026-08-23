from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ms_peso.service.backend import BackendPrediction
from ms_peso.service.uploads import StoredUpload


def build_prediction_payload(
    result: BackendPrediction,
    upload: StoredUpload,
    *,
    correlation_id: str | None,
) -> dict[str, object]:
    prediction = result.prediction
    descriptor = result.descriptor
    return {
        "schema_version": 1,
        "prediction_id": str(uuid4()),
        "correlation_id": correlation_id,
        "created_at": datetime.now(UTC).isoformat(),
        "prediction_status": "completed" if prediction else "rejected",
        "authorization_status": "blocked_pending_mandatory_reviews",
        "estimated_weight_kg": (
            round(prediction.estimated_weight_kg, 4) if prediction else None
        ),
        "prediction_interval": (
            {
                "lower_kg": round(prediction.interval_lower_kg, 4),
                "upper_kg": round(prediction.interval_upper_kg, 4),
                "radius_kg": round(prediction.interval_radius_kg, 4),
                "target_coverage": prediction.target_coverage,
                "method": "split_conformal_grouped_absolute_residual",
                "lower_bound_clipped_at_zero": (prediction.lower_bound_clipped_at_zero),
            }
            if prediction
            else None
        ),
        "quality_check_status": result.quality.status,
        "quality": result.quality.to_dict(),
        "input": {
            "filename": upload.original_filename,
            "content_type": upload.content_type,
            "size_bytes": upload.size_bytes,
            "original_width": result.quality.width,
            "original_height": result.quality.height,
            "expected_view": descriptor.input_view,
        },
        "model": {
            "id": descriptor.model_id,
            "version": descriptor.model_version,
            "architecture": descriptor.architecture,
            "checkpoint_sha256": descriptor.checkpoint_sha256,
            "calibration_sha256": descriptor.calibration_sha256,
            "evaluation_sha256": descriptor.evaluation_sha256,
            "quality_policy_sha256": descriptor.quality_policy_sha256,
            "model_card_sha256": descriptor.model_card_sha256,
            "status": descriptor.status,
            "production_ready": descriptor.production_ready,
            "commercial_use_allowed": descriptor.commercial_use_allowed,
            "commercial_blockers": list(descriptor.commercial_blockers),
            "training_dataset": descriptor.dataset,
            "training_breed": descriptor.breed,
            "device": result.device,
        },
        "warnings": list(descriptor.limitations),
    }
