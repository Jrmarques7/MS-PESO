import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ms_peso.importers.nellore_uav import (
    inspect_nellore_uav_dataset,
    scan_detection_annotations,
)
from ms_peso.inspect_nellore_uav import main


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=(80, 90, 100)).save(path)


def _write_labelme(path: Path, image_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "imagePath": image_name,
                "shapes": [],
            }
        ),
        encoding="utf-8",
    )


def create_fake_nellore_uav(root: Path) -> Path:
    dataset = root / "NelloreBeefCattleDataset"
    for session, count in (("10-07-2024", 2), ("31-07-2024", 1)):
        for index in range(count):
            _write_image(dataset / session / "15m" / f"DJI_{index:04d}.JPG")

    detection_image = (
        dataset
        / "annotations_v2"
        / "gado"
        / "images"
        / "train"
        / "2024-07-10"
        / "DJI_0000.JPG"
    )
    _write_image(detection_image)
    _write_labelme(detection_image.with_suffix(".json"), detection_image.name)
    detection_label = (
        dataset
        / "annotations_v2"
        / "gado"
        / "labels"
        / "train"
        / "2024-07-10"
        / "DJI_0000.txt"
    )
    detection_label.parent.mkdir(parents=True)
    detection_label.write_text(
        "0 0.5 0.5 0.2 0.3\n1 0.6 0.4 0.1 0.1\n", encoding="utf-8"
    )

    feed_bunk_image = (
        dataset / "annotations_v2" / "cocho" / "2024-07-10" / "DJI_0001.JPG"
    )
    _write_image(feed_bunk_image)
    _write_labelme(feed_bunk_image.with_suffix(".json"), feed_bunk_image.name)
    return dataset


def test_inspects_wrapped_dataset_without_claiming_weight_labels(tmp_path):
    dataset = create_fake_nellore_uav(tmp_path)
    inventory = inspect_nellore_uav_dataset(tmp_path)

    assert inventory.dataset_root == str(dataset.resolve())
    assert inventory.sessions == {"2024-07-10": 2, "2024-07-31": 1}
    assert inventory.raw_images == 3
    assert inventory.detection_images == 1
    assert inventory.detection_labelme_json == 1
    assert inventory.detection_yolo_labels == 1
    assert inventory.detection_classes == {"cattle-back": 1, "cattle-head": 1}
    assert inventory.feed_bunk_images == 1
    assert inventory.weight_metadata_files == ()
    assert inventory.regression_ready is False


def test_reports_candidate_weight_metadata_without_marking_regression_ready(tmp_path):
    dataset = create_fake_nellore_uav(tmp_path)
    (dataset / "peso_animais.csv").write_text("animal_id,peso\n1,400\n")

    inventory = inspect_nellore_uav_dataset(dataset)

    assert inventory.weight_metadata_files == ("peso_animais.csv",)
    assert inventory.regression_ready is False


def test_rejects_detection_image_without_yolo_label(tmp_path):
    dataset = create_fake_nellore_uav(tmp_path)
    label = next((dataset / "annotations_v2" / "gado" / "labels").rglob("*.txt"))
    label.unlink()

    with pytest.raises(ValueError, match="rótulo YOLO ausente"):
        scan_detection_annotations(dataset)


def test_rejects_session_without_15m_directory(tmp_path):
    dataset = create_fake_nellore_uav(tmp_path)
    invalid_session = dataset / "05-10-2024"
    invalid_session.mkdir()

    with pytest.raises(ValueError, match="Diretório 15m ausente"):
        inspect_nellore_uav_dataset(dataset)


def test_cli_writes_inventory_without_creating_regression_manifest(
    tmp_path, monkeypatch, capsys
):
    dataset = create_fake_nellore_uav(tmp_path)
    output = tmp_path / "inventory.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_nellore_uav",
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
        ],
    )

    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["raw_images"] == 3
    assert report["regression_ready"] is False
    assert "weight_kg" not in report
    assert "Apto para regressão de peso: False" in capsys.readouterr().out
