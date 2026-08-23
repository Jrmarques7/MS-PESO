import csv
from pathlib import Path

import pytest
from PIL import Image

from ms_peso.importers.mendeley_multiview import (
    build_multiview_manifest_rows,
    read_multiview_labels,
    read_multiview_metadata,
)
from ms_peso.manifest import validate_rows


def create_fake_multiview(root: Path, animals: int = 3) -> Path:
    dataset = root / "wrapper" / "Cow_Images"
    dataset.mkdir(parents=True)
    with (dataset / "labels.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["cow_id", "weight_kg"])
        writer.writeheader()
        for animal_id in range(1, animals + 1):
            writer.writerow({"cow_id": animal_id, "weight_kg": 300 + animal_id * 10})

    fields = [
        "image_name",
        "cow_id",
        "angle",
        "collection_date",
        "time_of_day",
        "weather",
        "camera_mode",
        "device",
        "collector_name",
        "gps_coordinates",
        "location",
    ]
    with (dataset / "metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for animal_id in range(1, animals + 1):
            for angle in range(1, 6):
                image_name = f"cow_{animal_id:04d}_angle{angle}.jpg"
                directory = dataset / f"Angle{angle}"
                directory.mkdir(exist_ok=True)
                Image.new("RGB", (32, 24), color=(animal_id * 20, angle, 40)).save(
                    directory / image_name
                )
                writer.writerow(
                    {
                        "image_name": image_name,
                        "cow_id": f"{animal_id:04d}",
                        "angle": angle,
                        "collection_date": "27-11-2025",
                        "time_of_day": "Morning",
                        "weather": "Sunny",
                        "camera_mode": "Auto",
                        "device": "Test Phone",
                        "collector_name": "Private Person",
                        "gps_coordinates": "1.234,5.678",
                        "location": "Test City",
                    }
                )
    return dataset


def test_reads_labels_and_complete_metadata(tmp_path: Path) -> None:
    create_fake_multiview(tmp_path)
    labels = read_multiview_labels(tmp_path)
    metadata = read_multiview_metadata(tmp_path)
    assert labels == {"1": "310", "2": "320", "3": "330"}
    assert len(metadata) == 15
    assert metadata[("1", "angle_1")]["device"] == "Test Phone"


def test_requires_explicit_acknowledgement_for_unverified_labels(
    tmp_path: Path,
) -> None:
    dataset = create_fake_multiview(tmp_path)
    with pytest.raises(ValueError, match="não foram validados independentemente"):
        build_multiview_manifest_rows(dataset, image_root=tmp_path)


def test_builds_review_manifest_without_personal_metadata(tmp_path: Path) -> None:
    dataset = create_fake_multiview(tmp_path)
    rows = build_multiview_manifest_rows(
        dataset,
        image_root=tmp_path,
        angles=["angle_1"],
        acknowledge_unverified_source_labels=True,
    )
    assert len(rows) == 3
    assert rows[0]["animal_id"] == "mendeley_multiview_0001"
    assert rows[0]["view"] == "angle_1"
    assert rows[0]["quality"] == "review_required"
    assert rows[0]["training_eligible"] == "false"
    assert "collector_name" not in rows[0]
    assert "gps_coordinates" not in rows[0]
    validate_rows(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=True,
    )


def test_rejects_incomplete_animal_angle_grid(tmp_path: Path) -> None:
    dataset = create_fake_multiview(tmp_path)
    metadata_path = dataset / "metadata.csv"
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    metadata_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="grade animal/ângulo"):
        build_multiview_manifest_rows(
            dataset,
            image_root=tmp_path,
            acknowledge_unverified_source_labels=True,
        )


def test_rejects_unknown_angle(tmp_path: Path) -> None:
    dataset = create_fake_multiview(tmp_path)
    with pytest.raises(ValueError, match="Ângulos inválidos"):
        build_multiview_manifest_rows(
            dataset,
            image_root=tmp_path,
            angles=["side"],
            acknowledge_unverified_source_labels=True,
        )
