import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ms_peso.artifacts import save_checkpoint, save_json
from ms_peso.collection_snapshot import calculate_image_dhash
from ms_peso.evaluate_commercial import main
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import COMMERCIAL_SPLITS, read_manifest, write_manifest


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 1),
    )


def _prepare_evaluation_case(tmp_path: Path) -> tuple[Path, Path]:
    assignments = ["train"] * 2 + ["val"] * 2 + ["calibration"] * 9 + ["test"] * 4
    rows: list[dict[str, str]] = []
    for index, split in enumerate(assignments):
        image_path = tmp_path / f"{split}_{index:02d}.png"
        Image.new(
            "RGB",
            (20, 20),
            (15 + index * 10, 225 - index * 6, 35 + index * 8),
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
    split_report_path = tmp_path / "split_report.json"
    save_json(
        split_report_path,
        {
            "status": "passed",
            "snapshot_id": "snapshot_evaluation_test",
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

    checkpoint_path = tmp_path / "fit_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        state_dict=_tiny_model().state_dict(),
        metadata={
            "architecture": "tiny_test_model",
            "dropout": 0.0,
            "target_mean": 300.0,
            "target_std": 50.0,
            "config": {
                "project": {"workflow": "commercial_fit", "seed": 42},
                "data": {"image_size": 16, "view": "left"},
                "model": {"pretrained": False, "initialization": "random"},
            },
            "workflow": "commercial_fit",
            "initialization": "random",
            "commercial_use_allowed": False,
            "promotion_status": "not_promoted",
            "held_out_partitions": ["calibration", "test"],
            "source_snapshot_id": "snapshot_evaluation_test",
            "source_manifest_sha256": calculate_sha256(manifest_path),
            "source_split_report_sha256": calculate_sha256(split_report_path),
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
            "architecture": "tiny_test_model",
            "image_size": 16,
            "input_view": "left",
            "target_coverage": 0.90,
            "interval_radius_kg": 1000.0,
            "quantile_rank": 9,
            "number_of_calibration_animals": 9,
            "test_evaluated": False,
            "promotion_status": "not_promoted",
            "commercial_use_allowed": False,
            "source_checkpoint_sha256": calculate_sha256(checkpoint_path),
            "source_snapshot_id": "snapshot_evaluation_test",
            "source_manifest_sha256": calculate_sha256(manifest_path),
            "source_split_report_sha256": calculate_sha256(split_report_path),
        },
    )
    config = {
        "project": {
            "name": "commercial-evaluation-test",
            "seed": 42,
            "workflow": "commercial_evaluation",
        },
        "data": {
            "manifest": str(manifest_path),
            "split_report": str(split_report_path),
            "image_root": str(tmp_path),
            "view": "left",
            "image_size": 16,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": calculate_sha256(checkpoint_path),
        },
        "calibration": {
            "report": str(calibration_path),
            "report_sha256": calculate_sha256(calibration_path),
        },
        "evaluation": {
            "bootstrap_iterations": 100,
            "access_receipt": str(tmp_path / "test_access.json"),
        },
        "acceptance": {
            "minimum_test_animals": 4,
            "maximum_mae_upper_95_kg": 1000.0,
            "maximum_rmse_upper_95_kg": 1000.0,
            "maximum_mape_upper_95_pct": 1000.0,
            "maximum_absolute_bias_bound_95_kg": 1000.0,
            "minimum_group_coverage_lower_95": 0.10,
            "maximum_interval_radius_kg": 2000.0,
        },
        "output": {"directory": str(tmp_path / "evaluation_output")},
    }
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, tmp_path / "test_access.json"


def test_final_evaluation_opens_only_test_and_consumes_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, receipt_path = _prepare_evaluation_case(tmp_path)
    opened_images: list[str] = []
    original_open = Image.open

    def tracked_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        opened_images.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(
        "ms_peso.evaluate_commercial.build_model",
        lambda *args, **kwargs: _tiny_model(),
    )
    monkeypatch.setattr("ms_peso.dataset.Image.open", tracked_open)
    monkeypatch.setattr("ms_peso.collection_snapshot.Image.open", tracked_open)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.evaluate_commercial",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
    )

    main()

    output_dir = tmp_path / "evaluation_output"
    assert receipt_path.is_file()
    assert (output_dir / "final_test_report.json").is_file()
    assert (output_dir / "predictions_test.csv").is_file()
    assert (output_dir / "resolved_test_manifest.csv").is_file()
    assert opened_images
    assert all(name.startswith("test_") for name in opened_images)
    rows = read_manifest(output_dir / "resolved_test_manifest.csv")
    assert {row["split"] for row in rows} == {"test"}
    report = json.loads((output_dir / "final_test_report.json").read_text("utf-8"))
    assert report["test_evaluated"] is True
    assert report["test_consumed"] is True
    assert report["technical_criteria_passed"] is True
    assert report["technical_recommendation"] == "technical_review_recommended"
    assert report["promotion_status"] == "review_required"
    assert report["commercial_use_allowed"] is False

    number_opened = len(opened_images)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.evaluate_commercial",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "different_output"),
        ],
    )
    with pytest.raises(FileExistsError, match="recibo de acesso"):
        main()
    assert len(opened_images) == number_opened


def test_wide_calibration_rejects_before_consuming_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, receipt_path = _prepare_evaluation_case(tmp_path)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    config["acceptance"]["maximum_interval_radius_kg"] = 100.0
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
        ["ms_peso.evaluate_commercial", "--config", str(config_path)],
    )

    with pytest.raises(ValueError, match="intervalo calibrado excede"):
        main()

    assert not receipt_path.exists()
    assert opened_images == []


def test_failed_metric_consumes_test_and_records_technical_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, receipt_path = _prepare_evaluation_case(tmp_path)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    config["acceptance"]["maximum_mae_upper_95_kg"] = 0.0001
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "ms_peso.evaluate_commercial.build_model",
        lambda *args, **kwargs: _tiny_model(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ms_peso.evaluate_commercial",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
    )

    main()

    report = json.loads(
        (tmp_path / "evaluation_output" / "final_test_report.json").read_text(
            "utf-8"
        )
    )
    assert receipt_path.is_file()
    assert report["test_consumed"] is True
    assert report["technical_criteria_passed"] is False
    assert report["technical_recommendation"] == "technical_rejection"
    assert report["promotion_status"] == "technical_rejected"
    assert report["commercial_use_allowed"] is False
