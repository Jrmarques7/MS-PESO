from pathlib import Path

from ms_peso.inference import WeightPrediction
from ms_peso.model_package import ModelDescriptor
from ms_peso.predict import build_prediction_payload, parse_args


def descriptor(tmp_path: Path) -> ModelDescriptor:
    return ModelDescriptor(
        model_id="b2-test",
        model_version="1",
        status="experimental",
        production_ready=False,
        architecture="efficientnet_b0",
        checkpoint_path=tmp_path / "model.pt",
        checkpoint_sha256="a" * 64,
        image_size=224,
        input_view="left",
        dataset="CowDB",
        breed="hereford",
        limitations=("Não validado para Nelore.",),
        model_card_path=tmp_path / "model-card.md",
    )


def test_cli_accepts_image_model_device_and_output() -> None:
    result = parse_args(
        [
            "--image",
            "cow.png",
            "--model",
            "model.yaml",
            "--device",
            "cpu",
            "--output",
            "prediction.json",
        ]
    )

    assert result.image == Path("cow.png")
    assert result.model == Path("model.yaml")
    assert result.device == "cpu"
    assert result.output == Path("prediction.json")


def test_payload_marks_experimental_model(tmp_path: Path) -> None:
    payload = build_prediction_payload(
        WeightPrediction(
            estimated_weight_kg=425.12345,
            original_width=100,
            original_height=50,
        ),
        descriptor(tmp_path),
        image_path=tmp_path / "cow.png",
        device="cpu",
        prediction_id="prediction-1",
        created_at="2026-08-22T12:00:00+00:00",
    )

    assert payload["estimated_weight_kg"] == 425.1234
    assert payload["quality_check_status"] == "not_implemented"
    assert payload["model"]["production_ready"] is False
    assert payload["warnings"] == ["Não validado para Nelore."]
