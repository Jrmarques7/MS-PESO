from pathlib import Path

import pytest
import yaml

from ms_peso.model_package import (
    calculate_sha256,
    load_model_descriptor,
    verify_model_package,
)


def write_descriptor(tmp_path: Path, checkpoint_sha256: str) -> Path:
    (tmp_path / "model_card.md").write_text("# Model card", encoding="utf-8")
    descriptor = {
        "schema_version": 1,
        "model_id": "test-model",
        "model_version": "1",
        "status": "experimental",
        "production_ready": False,
        "architecture": "efficientnet_b0",
        "checkpoint": {"path": "model.pt", "sha256": checkpoint_sha256},
        "input": {"image_size": 16, "view": "left"},
        "domain": {"dataset": "test", "breed": "hereford"},
        "limitations": ["Apenas teste."],
        "model_card": "model_card.md",
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return path


def test_loads_and_verifies_model_descriptor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    descriptor_path = write_descriptor(tmp_path, calculate_sha256(checkpoint))

    descriptor = load_model_descriptor(descriptor_path)
    verify_model_package(descriptor)

    assert descriptor.model_id == "test-model"
    assert descriptor.image_size == 16
    assert descriptor.checkpoint_path == checkpoint


def test_rejects_checkpoint_with_different_hash(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"checkpoint")
    descriptor = load_model_descriptor(write_descriptor(tmp_path, "0" * 64))

    with pytest.raises(ValueError, match="não corresponde"):
        verify_model_package(descriptor)


def test_rejects_invalid_sha256(tmp_path: Path) -> None:
    descriptor_path = write_descriptor(tmp_path, "invalid")

    with pytest.raises(ValueError, match="SHA-256"):
        load_model_descriptor(descriptor_path)
