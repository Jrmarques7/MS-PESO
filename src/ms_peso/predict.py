from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ms_peso.artifacts import save_json
from ms_peso.image_quality import (
    ImageQualityReport,
    assess_image_quality,
    load_image_quality_policy,
)
from ms_peso.inference import CattleWeightPredictor, WeightPrediction
from ms_peso.model_package import ModelDescriptor, load_model_descriptor


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima peso a partir de uma imagem RGB lateral."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="models/b2_cowdb.yaml", type=Path)
    parser.add_argument(
        "--quality-policy",
        default="configs/image_quality.yaml",
        type=Path,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def build_prediction_payload(
    prediction: WeightPrediction | None,
    descriptor: ModelDescriptor,
    quality_report: ImageQualityReport,
    *,
    image_path: Path,
    device: str | None,
    prediction_id: str,
    created_at: str,
) -> dict[str, object]:
    return {
        "prediction_id": prediction_id,
        "created_at": created_at,
        "prediction_status": "completed" if prediction is not None else "rejected",
        "estimated_weight_kg": (
            round(prediction.estimated_weight_kg, 4)
            if prediction is not None
            else None
        ),
        "quality_check_status": quality_report.status,
        "quality": quality_report.to_dict(),
        "input": {
            "image_path": str(image_path.resolve()),
            "original_width": quality_report.width,
            "original_height": quality_report.height,
            "expected_view": descriptor.input_view,
        },
        "model": {
            "id": descriptor.model_id,
            "version": descriptor.model_version,
            "architecture": descriptor.architecture,
            "checkpoint_sha256": descriptor.checkpoint_sha256,
            "status": descriptor.status,
            "production_ready": descriptor.production_ready,
            "training_dataset": descriptor.dataset,
            "training_breed": descriptor.breed,
            "device": device,
        },
        "warnings": list(descriptor.limitations),
    }


def main() -> None:
    args = parse_args()
    descriptor = load_model_descriptor(args.model)
    quality_policy = load_image_quality_policy(args.quality_policy)
    quality_report = assess_image_quality(args.image, quality_policy)
    prediction_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    if not quality_report.accepted:
        payload = build_prediction_payload(
            None,
            descriptor,
            quality_report,
            image_path=args.image,
            device=None,
            prediction_id=prediction_id,
            created_at=created_at,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    predictor = CattleWeightPredictor.load(descriptor, device=args.device)
    prediction = predictor.predict_image(args.image)
    payload = build_prediction_payload(
        prediction,
        descriptor,
        quality_report,
        image_path=args.image,
        device=str(predictor.device),
        prediction_id=prediction_id,
        created_at=created_at,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
