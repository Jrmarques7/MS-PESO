from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ms_peso.artifacts import save_json
from ms_peso.inference import CattleWeightPredictor, WeightPrediction
from ms_peso.model_package import ModelDescriptor, load_model_descriptor


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima peso a partir de uma imagem RGB lateral."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="models/b2_cowdb.yaml", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def build_prediction_payload(
    prediction: WeightPrediction,
    descriptor: ModelDescriptor,
    *,
    image_path: Path,
    device: str,
    prediction_id: str,
    created_at: str,
) -> dict[str, object]:
    return {
        "prediction_id": prediction_id,
        "created_at": created_at,
        "estimated_weight_kg": round(prediction.estimated_weight_kg, 4),
        "quality_check_status": "not_implemented",
        "input": {
            "image_path": str(image_path.resolve()),
            "original_width": prediction.original_width,
            "original_height": prediction.original_height,
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
    predictor = CattleWeightPredictor.load(descriptor, device=args.device)
    prediction = predictor.predict_image(args.image)
    payload = build_prediction_payload(
        prediction,
        descriptor,
        image_path=args.image,
        device=str(predictor.device),
        prediction_id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
