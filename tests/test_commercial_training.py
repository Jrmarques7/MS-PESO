import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ms_peso.artifacts import save_json
from ms_peso.collection_snapshot import calculate_image_dhash
from ms_peso.commercial_training import validate_commercial_fit_contract
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import COMMERCIAL_SPLITS, read_manifest, write_manifest
from ms_peso.train import main


def _prepare_commercial_input(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict[str, str]]]:
    assignments = (
        "train",
        "train",
        "train",
        "train",
        "val",
        "val",
        "calibration",
        "test",
    )
    rows: list[dict[str, str]] = []
    for index, split in enumerate(assignments):
        image_path = tmp_path / f"{split}_{index:02d}.png"
        Image.new(
            "RGB",
            (20, 20),
            (20 + index * 20, 210 - index * 10, 40 + index * 12),
        ).save(image_path)
        rows.append(
            {
                "image_path": image_path.name,
                "animal_id": f"animal_{index:02d}",
                "event_id": f"event_{index:02d}",
                "weight_kg": str(250 + index * 25),
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
            "snapshot_id": "snapshot_test_001",
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
    return manifest_path, report_path, rows


def _config(
    tmp_path: Path, manifest_path: Path, report_path: Path
) -> dict[str, object]:
    return {
        "project": {
            "name": "commercial-test",
            "seed": 42,
            "workflow": "commercial_fit",
        },
        "data": {
            "manifest": str(manifest_path),
            "split_report": str(report_path),
            "image_root": str(tmp_path),
            "view": "left",
            "image_size": 16,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "architecture": "tiny_test_model",
            "pretrained": False,
            "initialization": "random",
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "huber_delta": 1.0,
            "patience": 1,
        },
        "output": {"directory": str(tmp_path / "fit_output")},
    }


def test_contract_rejects_pretrained_weights(tmp_path: Path) -> None:
    manifest_path, report_path, rows = _prepare_commercial_input(tmp_path)
    config = _config(tmp_path, manifest_path, report_path)
    config["model"]["pretrained"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="pretrained: false"):
        validate_commercial_fit_contract(
            config,
            rows,
            manifest_path=manifest_path,
            split_report_path=report_path,
            output_dir=tmp_path / "fit_output",
        )


def test_contract_rejects_initial_checkpoint(tmp_path: Path) -> None:
    manifest_path, report_path, rows = _prepare_commercial_input(tmp_path)
    config = _config(tmp_path, manifest_path, report_path)
    config["model"]["checkpoint_path"] = "artifacts/b2/best_model.pt"  # type: ignore[index]

    with pytest.raises(ValueError, match="checkpoint_path"):
        validate_commercial_fit_contract(
            config,
            rows,
            manifest_path=manifest_path,
            split_report_path=report_path,
            output_dir=tmp_path / "fit_output",
        )


def test_contract_accepts_approved_four_way_split(tmp_path: Path) -> None:
    manifest_path, report_path, rows = _prepare_commercial_input(tmp_path)
    contract = validate_commercial_fit_contract(
        _config(tmp_path, manifest_path, report_path),
        rows,
        manifest_path=manifest_path,
        split_report_path=report_path,
        output_dir=tmp_path / "fit_output",
    )

    assert contract.snapshot_id == "snapshot_test_001"
    assert contract.split_report_sha256 == calculate_sha256(report_path)
    assert contract.number_of_images == {
        "train": 4,
        "val": 2,
        "calibration": 1,
        "test": 1,
    }


def test_commercial_fit_never_opens_calibration_or_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, report_path, _ = _prepare_commercial_input(tmp_path)
    config = _config(tmp_path, manifest_path, report_path)
    config_path = tmp_path / "commercial_fit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    opened_images: list[str] = []
    original_open = Image.open

    def tracked_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        opened_images.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    def tiny_model(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 1),
        )

    monkeypatch.setattr("ms_peso.train.build_model", tiny_model)
    monkeypatch.setattr("ms_peso.dataset.Image.open", tracked_open)
    monkeypatch.setattr("ms_peso.collection_snapshot.Image.open", tracked_open)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["ms_peso.train", "--config", str(config_path)])

    main()

    output_dir = tmp_path / "fit_output"
    assert (output_dir / "best_model.pt").is_file()
    assert (output_dir / "fit_metrics.json").is_file()
    assert (output_dir / "resolved_fit_manifest.csv").is_file()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "predictions_test.csv").exists()
    assert opened_images
    assert all(not name.startswith("calibration_") for name in opened_images)
    assert all(not name.startswith("test_") for name in opened_images)

    resolved_rows = read_manifest(output_dir / "resolved_fit_manifest.csv")
    assert {row["split"] for row in resolved_rows} == {"train", "val"}
    fit_report = json.loads((output_dir / "fit_metrics.json").read_text("utf-8"))
    assert fit_report["calibration_evaluated"] is False
    assert fit_report["test_evaluated"] is False
    assert fit_report["commercial_use_allowed"] is False

    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["initialization"] == "random"
    assert checkpoint["held_out_partitions"] == ["calibration", "test"]
    assert checkpoint["commercial_use_allowed"] is False
