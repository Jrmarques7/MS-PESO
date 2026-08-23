import json
import sys
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ms_peso.collection import (
    AuthorizationRecord,
    CollectionPolicy,
    audit_pilot_collection,
    read_authorization_registry,
)
from ms_peso.image_quality import ImageQualityPolicy
from ms_peso.validate_collection import main


def collection_policy() -> CollectionPolicy:
    return CollectionPolicy(
        policy_id="pilot-test",
        policy_version="1",
        required_columns=(
            "image_path",
            "animal_id",
            "event_id",
            "weight_kg",
            "view",
            "breed",
            "sex",
            "farm_id",
            "lot_id",
            "captured_at",
            "weighed_at",
            "camera_id",
            "scale_id",
            "quality",
            "scale_marker",
            "authorization_id",
            "commercial_training_allowed",
            "notes",
        ),
        allowed_views=("left",),
        allowed_breeds=("nelore",),
        allowed_sexes=("male", "female"),
        min_weight_kg=100,
        max_weight_kg=1000,
        max_capture_weight_delta_minutes=30,
        required_quality="accepted",
        require_scale_marker=True,
        require_commercial_training_rights=True,
    )


def quality_policy() -> ImageQualityPolicy:
    return ImageQualityPolicy(
        policy_id="quality-test",
        policy_version="1",
        analysis_max_dimension=512,
        min_width=480,
        min_height=320,
        min_aspect_ratio=1.15,
        max_aspect_ratio=2,
        min_mean_luma=35,
        max_mean_luma=220,
        max_dark_fraction=0.8,
        max_bright_fraction=0.8,
        min_sharpness=0,
        limitations=("Não valida pose.",),
    )


def authorization(*, commercial: bool = True) -> AuthorizationRecord:
    return AuthorizationRecord(
        authorization_id="auth_001",
        farm_id="farm_001",
        status="approved",
        effective_from=date(2026, 8, 1),
        effective_until=None,
        allows_model_training=True,
        allows_commercial_use=commercial,
        allows_data_sharing=False,
        document_reference="secure://auth_001",
    )


def row(**updates: str) -> dict[str, str]:
    result = {
        "image_path": "cow.jpg",
        "animal_id": "nelore_001",
        "event_id": "event_001",
        "weight_kg": "425.5",
        "view": "left",
        "breed": "nelore",
        "sex": "female",
        "farm_id": "farm_001",
        "lot_id": "lot_001",
        "captured_at": "2026-08-22T09:02:00-03:00",
        "weighed_at": "2026-08-22T09:00:00-03:00",
        "camera_id": "camera_001",
        "scale_id": "scale_001",
        "quality": "accepted",
        "scale_marker": "true",
        "authorization_id": "auth_001",
        "commercial_training_allowed": "true",
        "notes": "",
    }
    result.update(updates)
    return result


def test_accepts_complete_commercial_collection_record(tmp_path: Path) -> None:
    Image.new("RGB", (800, 500), (128, 128, 128)).save(tmp_path / "cow.jpg")

    report = audit_pilot_collection(
        [row()],
        {"auth_001": authorization()},
        collection_policy(),
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=True,
        quality_policy=quality_policy(),
    )

    assert report.valid is True
    assert report.technical_quality_passed == 1
    assert report.to_dict()["status"] == "passed"


def test_rejects_missing_rights_marker_and_distant_weight(tmp_path: Path) -> None:
    invalid_row = row(
        captured_at="2026-08-22T11:00:00-03:00",
        scale_marker="false",
        commercial_training_allowed="false",
    )

    report = audit_pilot_collection(
        [invalid_row],
        {"auth_001": authorization(commercial=False)},
        collection_policy(),
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=False,
    )
    errors = "\n".join(report.errors)

    assert report.valid is False
    assert "marcador de escala" in errors
    assert "não autoriza treinamento comercial" in errors
    assert "separadas por 120.0 minutos" in errors
    assert "não cobre treinamento e uso comercial" in errors


def test_rejects_two_selected_images_for_same_event_view(tmp_path: Path) -> None:
    second_row = row(image_path="cow_2.jpg")

    report = audit_pilot_collection(
        [row(), second_row],
        {"auth_001": authorization()},
        collection_policy(),
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=False,
    )

    assert any("mais de uma imagem selecionada" in error for error in report.errors)


def test_rejects_invalid_authorization_registry(tmp_path: Path) -> None:
    registry = tmp_path / "authorizations.csv"
    registry.write_text(
        """authorization_id,farm_id,status,effective_from,effective_until,allows_model_training,allows_commercial_use,allows_data_sharing,document_reference,notes
auth_001,farm_001,approved,2026-08-01,,true,maybe,false,secure://auth_001,
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allows_commercial_use"):
        read_authorization_registry(registry)


def _write_checkerboard(path: Path) -> None:
    image = Image.new("RGB", (800, 500), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    for y in range(0, 500, 16):
        for x in range(0, 800, 16):
            if (x // 16 + y // 16) % 2 == 0:
                draw.rectangle((x, y, x + 15, y + 15), fill=(224, 224, 224))
    image.save(path)


def test_cli_validates_kit_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_checkerboard(tmp_path / "cow.jpg")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        ",".join(row().keys()) + "\n" + ",".join(row().values()) + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "authorizations.csv"
    registry.write_text(
        """authorization_id,farm_id,status,effective_from,effective_until,allows_model_training,allows_commercial_use,allows_data_sharing,document_reference,notes
auth_001,farm_001,approved,2026-08-01,,true,true,false,secure://auth_001,
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.validate_collection",
            "--manifest",
            str(manifest),
            "--authorizations",
            str(registry),
            "--image-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["summary"]["technical_quality_passed"] == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payload
