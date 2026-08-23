import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ms_peso.artifacts import save_checkpoint, save_json
from ms_peso.calibrate import main
from ms_peso.collection_snapshot import calculate_image_dhash
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import COMMERCIAL_SPLITS, read_manifest, write_manifest


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 1),
    )


def _prepare_calibration_case(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    assignments = ["train"] * 2 + ["val"] * 2 + ["calibration"] * 9 + ["test"] * 2
    rows: list[dict[str, str]] = []
    for index, split in enumerate(assignments):
        image_path = tmp_path / f"{split}_{index:02d}.png"
        Image.new(
            "RGB",
            (20, 20),
            (10 + index * 11, 220 - index * 7, 30 + index * 9),
        ).save(image_path)
        rows.append(
            {
                "image_path": image_path.name,
                "animal_id": f"animal_{index:02d}",
                "event_id": f"event_{index:02d}",
                "weight_kg": str(250 + index * 10),
                "view": "left",
                "split": split,
                "image_sha256": calculate_sha256(image_path),
                "image_dhash": calculate_image_dhash(image_path),
            }
        )

    manifest_path = tmp_path / "commercial.csv"
    write_manifest(rows, manifest_path)
    image_counts = Counter(row["split"] for row in rows)
    report_path = tmp_path / "split_report.json"
    save_json(
        report_path,
        {
            "status": "passed",
            "snapshot_id": "snapshot_calibration_test",
            "splits": {
                split: {
                    "animals": len(
                        {
                            row["animal_id"]
                            for row in rows
                            if row["split"] == split
                        }
                    ),
                    "images": image_counts[split],
                }
                for split in COMMERCIAL_SPLITS
            },
            "provenance": {
                "output_manifest_sha256": calculate_sha256(manifest_path)
            },
        },
    )

    fit_config = {
        "project": {"workflow": "commercial_fit", "seed": 42},
        "data": {"image_size": 16, "view": "left"},
        "model": {
            "architecture": "tiny_test_model",
            "pretrained": False,
            "initialization": "random",
            "dropout": 0.0,
        },
    }
    checkpoint_path = tmp_path / "fit_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        state_dict=_tiny_model().state_dict(),
        metadata={
            "architecture": "tiny_test_model",
            "dropout": 0.0,
            "target_mean": 300.0,
            "target_std": 50.0,
            "config": fit_config,
            "epoch": 1,
            "workflow": "commercial_fit",
            "initialization": "random",
            "commercial_use_allowed": False,
            "promotion_status": "not_promoted",
            "held_out_partitions": ["calibration", "test"],
            "source_snapshot_id": "snapshot_calibration_test",
            "source_manifest_sha256": calculate_sha256(manifest_path),
            "source_split_report_sha256": calculate_sha256(report_path),
        },
    )

    config = {
        "project": {
            "name": "commercial-calibration-test",
            "seed": 42,
            "workflow": "commercial_calibration",
        },
        "data": {
            "manifest": str(manifest_path),
            "split_report": str(report_path),
            "image_root": str(tmp_path),
            "view": "left",
            "image_size": 16,
            "batch_size": 3,
            "num_workers": 0,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": calculate_sha256(checkpoint_path),
        },
        "calibration": {"target_coverage": 0.90},
        "output": {"directory": str(tmp_path / "calibration_output")},
    }
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return manifest_path, checkpoint_path, config_path, report_path


def test_calibration_opens_only_reserved_calibration_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, config_path, _ = _prepare_calibration_case(tmp_path)
    opened_images: list[str] = []
    original_open = Image.open

    def tracked_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        opened_images.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(
        "ms_peso.calibrate.build_model", lambda *args, **kwargs: _tiny_model()
    )
    monkeypatch.setattr("ms_peso.dataset.Image.open", tracked_open)
    monkeypatch.setattr("ms_peso.collection_snapshot.Image.open", tracked_open)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ms_peso.calibrate", "--config", str(config_path), "--device", "cpu"],
    )

    main()

    output_dir = tmp_path / "calibration_output"
    assert (output_dir / "calibration.json").is_file()
    assert (output_dir / "predictions_calibration.csv").is_file()
    assert (output_dir / "resolved_calibration_manifest.csv").is_file()
    assert not (output_dir / "predictions_test.csv").exists()
    assert opened_images
    assert all(name.startswith("calibration_") for name in opened_images)

    rows = read_manifest(output_dir / "resolved_calibration_manifest.csv")
    assert {row["split"] for row in rows} == {"calibration"}
    report = json.loads((output_dir / "calibration.json").read_text("utf-8"))
    assert report["test_evaluated"] is False
    assert report["number_of_calibration_animals"] == 9
    assert report["target_coverage"] == pytest.approx(0.90)
    assert math.isfinite(report["interval_radius_kg"])
    assert report["commercial_use_allowed"] is False


def test_calibration_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, checkpoint_path, config_path, _ = _prepare_calibration_case(tmp_path)
    with checkpoint_path.open("ab") as file:
        file.write(b"changed")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ms_peso.calibrate", "--config", str(config_path), "--device", "cpu"],
    )

    with pytest.raises(ValueError, match="Hash do checkpoint diverge"):
        main()


def test_insufficient_animals_fails_before_opening_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, config_path, _ = _prepare_calibration_case(tmp_path)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    config["calibration"]["target_coverage"] = 0.95
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    opened_images: list[str] = []
    original_open = Image.open

    def tracked_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        opened_images.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("ms_peso.dataset.Image.open", tracked_open)
    monkeypatch.setattr("ms_peso.collection_snapshot.Image.open", tracked_open)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ms_peso.calibrate", "--config", str(config_path), "--device", "cpu"],
    )

    with pytest.raises(ValueError, match="Animais de calibração insuficientes"):
        main()

    assert opened_images == []
