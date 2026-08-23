import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ms_peso.artifacts import save_checkpoint, save_json
from ms_peso.commercial_inference import CommercialCandidatePredictor
from ms_peso.commercial_model_package import (
    load_commercial_candidate_descriptor,
    verify_commercial_candidate_package,
)
from ms_peso.integrity import calculate_sha256
from ms_peso.predict_commercial import main


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.5]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(inputs.shape[0], 1)


def _prepare_candidate_package(tmp_path: Path) -> Path:
    source_manifest_sha256 = "a" * 64
    source_split_report_sha256 = "b" * 64
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        state_dict=TinyRegressor().state_dict(),
        metadata={
            "architecture": "efficientnet_b0",
            "dropout": 0.2,
            "target_mean": 400.0,
            "target_std": 20.0,
            "config": {
                "project": {"workflow": "commercial_fit"},
                "model": {"pretrained": False, "initialization": "random"},
                "data": {"image_size": 16, "view": "left"},
            },
            "workflow": "commercial_fit",
            "initialization": "random",
            "commercial_use_allowed": False,
            "promotion_status": "not_promoted",
            "source_snapshot_id": "snapshot_candidate_test",
            "source_manifest_sha256": source_manifest_sha256,
            "source_split_report_sha256": source_split_report_sha256,
        },
    )
    calibration_path = tmp_path / "calibration.json"
    save_json(
        calibration_path,
        {
            "status": "calibrated",
            "workflow": "commercial_calibration",
            "method": "split_conformal_grouped_absolute_residual",
            "grouping": "animal_id_max_absolute_residual",
            "architecture": "efficientnet_b0",
            "image_size": 16,
            "input_view": "left",
            "target_coverage": 0.90,
            "interval_radius_kg": 20.0,
            "test_evaluated": False,
            "promotion_status": "not_promoted",
            "commercial_use_allowed": False,
            "source_checkpoint_sha256": calculate_sha256(checkpoint_path),
            "source_snapshot_id": "snapshot_candidate_test",
            "source_manifest_sha256": source_manifest_sha256,
            "source_split_report_sha256": source_split_report_sha256,
        },
    )
    evaluation_path = tmp_path / "evaluation.json"
    save_json(
        evaluation_path,
        {
            "status": "evaluated",
            "workflow": "commercial_evaluation",
            "test_evaluated": True,
            "test_consumed": True,
            "architecture": "efficientnet_b0",
            "image_size": 16,
            "input_view": "left",
            "technical_criteria_passed": True,
            "technical_recommendation": "technical_review_recommended",
            "promotion_status": "review_required",
            "commercial_use_allowed": False,
            "source_checkpoint_sha256": calculate_sha256(checkpoint_path),
            "source_calibration_sha256": calculate_sha256(calibration_path),
            "source_snapshot_id": "snapshot_candidate_test",
            "source_manifest_sha256": source_manifest_sha256,
            "source_split_report_sha256": source_split_report_sha256,
            "interval_method": "split_conformal_grouped_absolute_residual",
            "interval_target_coverage": 0.90,
            "interval_coverage": {"radius_kg": 20.0},
            "mandatory_remaining_reviews": [
                "legal_rights_review",
                "external_domain_validation",
                "operational_safety_review",
                "human_promotion_approval",
            ],
        },
    )
    quality_policy_path = tmp_path / "quality.yaml"
    quality_policy_path.write_text(
        """
schema_version: 1
policy_id: candidate-test-policy
policy_version: "1"
analysis:
  max_dimension: 64
thresholds:
  min_width: 16
  min_height: 12
  min_aspect_ratio: 0.5
  max_aspect_ratio: 3.0
  min_mean_luma: 1
  max_mean_luma: 254
  max_dark_fraction: 0.95
  max_bright_fraction: 0.95
  min_sharpness: 0
limitations:
  - Não valida pose.
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "model_card.md").write_text("# Candidate", encoding="utf-8")
    descriptor = {
        "schema_version": 1,
        "model_id": "candidate-test",
        "model_version": "1",
        "status": "candidate_unapproved",
        "production_ready": False,
        "commercial_use_allowed": False,
        "commercial_blockers": ["Aprovação final pendente."],
        "architecture": "efficientnet_b0",
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": calculate_sha256(checkpoint_path),
        },
        "calibration": {
            "path": calibration_path.name,
            "sha256": calculate_sha256(calibration_path),
        },
        "evaluation": {
            "path": evaluation_path.name,
            "sha256": calculate_sha256(evaluation_path),
        },
        "quality_policy": {
            "path": quality_policy_path.name,
            "sha256": calculate_sha256(quality_policy_path),
        },
        "input": {"view": "left", "image_size": 16},
        "domain": {"dataset": "owned-pilot", "breed": "nelore"},
        "limitations": ["Somente teste interno."],
        "model_card": {
            "path": "model_card.md",
            "sha256": calculate_sha256(tmp_path / "model_card.md"),
        },
    }
    descriptor_path = tmp_path / "candidate.yaml"
    descriptor_path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return descriptor_path


def test_loads_verified_candidate_and_predicts_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_path = _prepare_candidate_package(tmp_path)
    descriptor = load_commercial_candidate_descriptor(descriptor_path)
    monkeypatch.setattr(
        "ms_peso.commercial_inference.build_model",
        lambda *args, **kwargs: TinyRegressor(),
    )
    image_path = tmp_path / "cow.png"
    Image.new("RGB", (32, 24), (100, 120, 140)).save(image_path)

    predictor = CommercialCandidatePredictor.load(descriptor, device="cpu")
    prediction = predictor.predict_image(image_path)

    assert prediction.estimated_weight_kg == pytest.approx(410.0)
    assert prediction.interval_lower_kg == pytest.approx(390.0)
    assert prediction.interval_upper_kg == pytest.approx(430.0)
    assert prediction.target_coverage == pytest.approx(0.90)
    assert descriptor.commercial_use_allowed is False


def test_descriptor_rejects_commercial_authorization(tmp_path: Path) -> None:
    descriptor_path = _prepare_candidate_package(tmp_path)
    payload = yaml.safe_load(descriptor_path.read_text("utf-8"))
    payload["commercial_use_allowed"] = True
    descriptor_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="não pode declarar liberação"):
        load_commercial_candidate_descriptor(descriptor_path)


def test_package_rejects_tampered_evaluation(tmp_path: Path) -> None:
    descriptor_path = _prepare_candidate_package(tmp_path)
    descriptor = load_commercial_candidate_descriptor(descriptor_path)
    with descriptor.evaluation_path.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(ValueError, match="avaliação final"):
        verify_commercial_candidate_package(descriptor)


def test_cli_rejects_quality_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor_path = _prepare_candidate_package(tmp_path)
    image_path = tmp_path / "small.png"
    Image.new("RGB", (4, 4), (100, 120, 140)).save(image_path)

    def fail_if_model_loads(*args: object, **kwargs: object) -> None:
        pytest.fail("O checkpoint não deve ser carregado para imagem rejeitada.")

    monkeypatch.setattr(
        "ms_peso.predict_commercial.CommercialCandidatePredictor.load",
        fail_if_model_loads,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.predict_commercial",
            "--image",
            str(image_path),
            "--package",
            str(descriptor_path),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    payload = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 2
    assert payload["prediction_status"] == "rejected"
    assert payload["estimated_weight_kg"] is None
    assert payload["prediction_interval"] is None
    assert payload["model"]["commercial_use_allowed"] is False


def test_cli_returns_weight_interval_and_explicit_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor_path = _prepare_candidate_package(tmp_path)
    image_path = tmp_path / "cow.png"
    Image.new("RGB", (32, 24), (100, 120, 140)).save(image_path)
    monkeypatch.setattr(
        "ms_peso.commercial_inference.build_model",
        lambda *args, **kwargs: TinyRegressor(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.predict_commercial",
            "--image",
            str(image_path),
            "--package",
            str(descriptor_path),
            "--device",
            "cpu",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["prediction_status"] == "completed"
    assert payload["estimated_weight_kg"] == pytest.approx(410.0)
    assert payload["prediction_interval"]["lower_kg"] == pytest.approx(390.0)
    assert payload["prediction_interval"]["upper_kg"] == pytest.approx(430.0)
    assert payload["authorization_status"] == "blocked_pending_mandatory_reviews"
    assert payload["model"]["commercial_use_allowed"] is False
