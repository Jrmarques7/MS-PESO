from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image

from ms_peso.importers.cowdb import (
    build_cowdb_manifest_rows,
    read_cowdb_measurements,
    scan_cowdb_rgb_images,
)
from ms_peso.manifest import validate_rows

HEADERS = [
    "N",
    "live weithg",
    "withers height",
    "hip height",
    "chest depth",
    "chest width",
    "ilium width",
    "hip joint width",
    "oblique body length",
    "hip length",
    "heart girth",
]


def create_fake_cowdb(root: Path, number_of_animals: int = 3) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    for animal_id in range(1, number_of_animals + 1):
        sheet.append(
            [
                animal_id,
                300 + animal_id * 20,
                120,
                125,
                60,
                42,
                44,
                46,
                175,
                45,
                180,
            ]
        )
        for view in ("left", "right", "top"):
            directory = root / str(animal_id) / "raw" / view
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 24), color=(animal_id * 20, 30, 40)).save(
                directory / f"rgb-12.00.0{animal_id}.png"
            )
            Image.new("I;16", (16, 12), color=4000).save(
                directory / f"depth-12.00.0{animal_id}.png"
            )
    measurements = root / "Manual_measurements.xlsx"
    workbook.save(measurements)
    return measurements


def test_reads_original_cowdb_header_typo(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    measurements = read_cowdb_measurements(measurements_path)
    assert measurements["1"]["weight_kg"] == "320"
    assert measurements["1"]["heart_girth_cm"] == "180"


def test_builds_manifest_for_selected_view(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    rows = build_cowdb_manifest_rows(
        dataset_root,
        measurements_path,
        image_root=tmp_path,
        views=["left"],
    )
    assert len(rows) == 3
    assert rows[0]["animal_id"] == "cowdb_001"
    assert rows[0]["event_id"] == "cowdb_001_capture_001"
    assert rows[0]["image_path"].startswith("cowdb/1/raw/left/rgb-")
    assert rows[0]["breed"] == "hereford"
    assert rows[0]["weight_kg"] == "320"
    validate_rows(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=True,
    )


def test_multiple_views_share_event_and_animal(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    rows = build_cowdb_manifest_rows(
        dataset_root,
        measurements_path,
        image_root=tmp_path,
        views=["left", "top"],
    )
    animal_rows = [row for row in rows if row["animal_id"] == "cowdb_001"]
    assert {row["view"] for row in animal_rows} == {"left", "top"}
    assert len({row["event_id"] for row in animal_rows}) == 1


def test_includes_synchronized_depth_path(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    rows = build_cowdb_manifest_rows(
        dataset_root,
        measurements_path,
        image_root=tmp_path,
        views=["left"],
        include_depth=True,
    )

    assert rows[0]["depth_image_path"].endswith("depth-12.00.01.png")
    validate_rows(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=True,
        additional_image_columns=("depth_image_path",),
    )


def test_rejects_missing_synchronized_depth(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    next((dataset_root / "2" / "raw" / "left").glob("depth-*.png")).unlink()

    with pytest.raises(ValueError, match="Profundidade pareada ausente"):
        build_cowdb_manifest_rows(
            dataset_root,
            measurements_path,
            image_root=tmp_path,
            views=["left"],
            include_depth=True,
        )


def test_rejects_unknown_view(tmp_path):
    dataset_root = tmp_path / "cowdb"
    create_fake_cowdb(dataset_root)
    with pytest.raises(ValueError, match="Vistas CowDB inválidas"):
        scan_cowdb_rgb_images(dataset_root, views=["rear"])


def test_rejects_measurement_without_selected_image(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    missing_image = next((dataset_root / "3" / "raw" / "left").glob("rgb-*.png"))
    missing_image.unlink()
    with pytest.raises(ValueError, match="Medidas sem imagens"):
        build_cowdb_manifest_rows(
            dataset_root,
            measurements_path,
            image_root=tmp_path,
            views=["left"],
        )


def test_rejects_partially_missing_view(tmp_path):
    dataset_root = tmp_path / "cowdb"
    measurements_path = create_fake_cowdb(dataset_root)
    missing_image = next((dataset_root / "2" / "raw" / "top").glob("rgb-*.png"))
    missing_image.unlink()
    with pytest.raises(ValueError, match="Vistas RGB ausentes"):
        build_cowdb_manifest_rows(
            dataset_root,
            measurements_path,
            image_root=tmp_path,
            views=["left", "top"],
        )
