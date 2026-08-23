from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image

from ms_peso.importers.horqin import (
    build_horqin_manifest_rows,
    find_horqin_anomalies,
    read_horqin_measurements,
    scan_horqin_images,
)
from ms_peso.manifest import grouped_split, validate_rows

HEADERS = [
    "Num",
    "Oblique body length (cm)",
    "Withers height(cm)",
    "Heart girth(cm)",
    "Hip length (cm)",
    "Body weight (kg)",
]


def create_fake_horqin(root: Path, animals: int = 4) -> Path:
    dataset = root / "wrapper" / "Cattle side and back view images"
    (dataset / "side view").mkdir(parents=True)
    (dataset / "back view").mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    for animal_id in range(1, animals + 1):
        sheet.append([animal_id, 150, 120, 180, 44, 400 + animal_id * 10])
        Image.new("RGB", (32, 24), color=(animal_id * 20, 30, 40)).save(
            dataset / "side view" / f"{animal_id}.png"
        )
        Image.new("RGB", (24, 32), color=(animal_id * 20, 40, 50)).save(
            dataset / "back view" / f"{animal_id}.png"
        )
    workbook.save(dataset / "measurements.xlsx")
    return dataset


def test_reads_measurements_and_images(tmp_path: Path) -> None:
    create_fake_horqin(tmp_path)
    measurements = read_horqin_measurements(tmp_path)
    images = scan_horqin_images(tmp_path)
    assert measurements["1"]["weight_kg"] == "410"
    assert measurements["1"]["heart_girth_cm"] == "180"
    assert set(images["1"]) == {"side", "back"}


def test_builds_side_manifest_and_preserves_animal_split(tmp_path: Path) -> None:
    dataset = create_fake_horqin(tmp_path)
    rows = build_horqin_manifest_rows(
        dataset,
        image_root=tmp_path,
        views=["side"],
        min_short_edge_px=20,
    )
    assert len(rows) == 4
    assert rows[0]["animal_id"] == "horqin_001"
    assert rows[0]["view"] == "side"
    assert rows[0]["breed"] == "horqin"
    assert rows[0]["source_license"] == "CC_BY_4_0"
    validate_rows(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        check_images=True,
    )
    split_rows = grouped_split(rows, seed=42)
    assert len({row["split"] for row in split_rows}) == 3


def test_requires_explicit_exclusion_for_missing_view(tmp_path: Path) -> None:
    dataset = create_fake_horqin(tmp_path)
    (dataset / "side view" / "2.png").unlink()
    with pytest.raises(ValueError, match="exclude_known_anomalies=True"):
        build_horqin_manifest_rows(
            dataset,
            image_root=tmp_path,
            views=["side"],
            min_short_edge_px=20,
        )

    rows = build_horqin_manifest_rows(
        dataset,
        image_root=tmp_path,
        views=["side"],
        exclude_known_anomalies=True,
        min_short_edge_px=20,
    )
    assert {row["source_animal_id"] for row in rows} == {"1", "3", "4"}


def test_finds_and_excludes_low_resolution_view(tmp_path: Path) -> None:
    dataset = create_fake_horqin(tmp_path)
    Image.new("RGB", (10, 10)).save(dataset / "back view" / "3.png")
    anomalies = find_horqin_anomalies(
        dataset, views=["back"], min_short_edge_px=20
    )
    assert anomalies == {"3": ("low_resolution_back:10x10",)}
    rows = build_horqin_manifest_rows(
        dataset,
        image_root=tmp_path,
        views=["side", "back"],
        exclude_known_anomalies=True,
        min_short_edge_px=20,
    )
    assert len(rows) == 6
    assert "horqin_003" not in {row["animal_id"] for row in rows}


def test_rejects_unknown_view(tmp_path: Path) -> None:
    dataset = create_fake_horqin(tmp_path)
    with pytest.raises(ValueError, match="Vistas Horqin inválidas"):
        build_horqin_manifest_rows(
            dataset,
            image_root=tmp_path,
            views=["top"],
            min_short_edge_px=20,
        )
