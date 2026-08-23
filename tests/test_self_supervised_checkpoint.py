from pathlib import Path

import pytest
import torch

from ms_peso.artifacts import save_checkpoint
from ms_peso.integrity import calculate_sha256
from ms_peso.self_supervised import load_self_supervised_encoder


def create_checkpoint(path: Path, **overrides) -> str:
    metadata = {
        "checkpoint_type": "self_supervised_encoder",
        "architecture": "efficientnet_b0",
        "pretrained": False,
        "initialization": "random",
        "labels_used": False,
        "commercial_use_allowed": False,
        "promotion_status": "not_promoted",
        "source_license": "CC_BY_4_0",
        "source_manifest_sha256": "a" * 64,
        **overrides,
    }
    save_checkpoint(path, state_dict={"weight": torch.ones(1)}, metadata=metadata)
    return calculate_sha256(path)


def test_loads_verified_self_supervised_encoder(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    digest = create_checkpoint(path)
    checkpoint = load_self_supervised_encoder(path, expected_sha256=digest)
    assert checkpoint.architecture == "efficientnet_b0"
    assert checkpoint.source_manifest_sha256 == "a" * 64


def test_rejects_encoder_that_used_labels(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    digest = create_checkpoint(path, labels_used=True)
    with pytest.raises(ValueError, match="labels_used"):
        load_self_supervised_encoder(path, expected_sha256=digest)


def test_rejects_encoder_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    create_checkpoint(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_self_supervised_encoder(path, expected_sha256="b" * 64)
