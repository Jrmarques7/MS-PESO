import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ms_peso.image_quality import ImageQualityReport, QualityCheck
from ms_peso.inference import WeightPrediction
from ms_peso.model_package import ModelDescriptor
from ms_peso.predict import build_prediction_payload, main, parse_args


def descriptor(tmp_path: Path) -> ModelDescriptor:
    return ModelDescriptor(
        model_id="b2-test",
        model_version="1",
        status="experimental",
        production_ready=False,
        commercial_use_allowed=False,
        commercial_blockers=("Apenas pesquisa.",),
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


def quality_report(*, accepted: bool = True) -> ImageQualityReport:
    return ImageQualityReport(
        policy_id="test-policy",
        policy_version="1",
        width=100,
        height=50,
        checks=(
            QualityCheck(
                code="test",
                passed=accepted,
                value=1,
                requirement=">= 1",
                rejection_message="Imagem rejeitada.",
            ),
        ),
        limitations=("Não valida pose.",),
    )


def test_cli_accepts_image_model_device_and_output() -> None:
    result = parse_args(
        [
            "--image",
            "cow.png",
            "--model",
            "model.yaml",
            "--quality-policy",
            "quality.yaml",
            "--device",
            "cpu",
            "--output",
            "prediction.json",
        ]
    )

    assert result.image == Path("cow.png")
    assert result.model == Path("model.yaml")
    assert result.quality_policy == Path("quality.yaml")
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
        quality_report(),
        image_path=tmp_path / "cow.png",
        device="cpu",
        prediction_id="prediction-1",
        created_at="2026-08-22T12:00:00+00:00",
    )

    assert payload["estimated_weight_kg"] == 425.1234
    assert payload["prediction_status"] == "completed"
    assert payload["quality_check_status"] == "passed"
    assert payload["quality"]["rejection_reasons"] == []
    assert payload["model"]["production_ready"] is False
    assert payload["model"]["commercial_use_allowed"] is False
    assert payload["warnings"] == ["Não validado para Nelore."]


def test_payload_omits_estimate_when_quality_is_rejected(tmp_path: Path) -> None:
    payload = build_prediction_payload(
        None,
        descriptor(tmp_path),
        quality_report(accepted=False),
        image_path=tmp_path / "cow.png",
        device=None,
        prediction_id="prediction-2",
        created_at="2026-08-22T12:00:00+00:00",
    )

    assert payload["prediction_status"] == "rejected"
    assert payload["estimated_weight_kg"] is None
    assert payload["quality_check_status"] == "rejected"
    assert payload["quality"]["rejection_reasons"] == ["Imagem rejeitada."]
    assert payload["model"]["device"] is None


def test_cli_rejects_bad_image_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image_path = tmp_path / "dark.png"
    Image.new("RGB", (800, 500), (0, 0, 0)).save(image_path)
    descriptor_path = tmp_path / "model.yaml"
    descriptor_path.write_text(
        f"""
schema_version: 1
model_id: b2-test
model_version: "1"
status: experimental
production_ready: false
commercial_use_allowed: false
commercial_blockers:
  - Apenas pesquisa.
architecture: efficientnet_b0
checkpoint:
  path: model.pt
  sha256: {"a" * 64}
input:
  view: left
  image_size: 224
domain:
  dataset: CowDB
  breed: hereford
limitations:
  - Não validado para Nelore.
model_card: model-card.md
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "rejection.json"

    def fail_if_model_loads(*args: object, **kwargs: object) -> None:
        pytest.fail("O modelo não deve ser carregado para uma imagem rejeitada.")

    monkeypatch.setattr(
        "ms_peso.predict.CattleWeightPredictor.load", fail_if_model_loads
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.predict",
            "--image",
            str(image_path),
            "--model",
            str(descriptor_path),
            "--quality-policy",
            "configs/image_quality.yaml",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    payload = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 2
    assert payload["prediction_status"] == "rejected"
    assert payload["estimated_weight_kg"] is None
    assert output_path.is_file()
