import csv

import pytest

from ms_peso.manifest import (
    assert_no_animal_leakage,
    grouped_split,
    read_manifest,
    validate_rows,
    write_manifest,
)


def make_rows(number_of_animals: int = 30):
    return [
        {
            "image_path": f"images/cow_{index:03d}.jpg",
            "animal_id": f"cow_{index:03d}",
            "event_id": f"event_{index:03d}",
            "weight_kg": str(200 + index * 10),
            "view": "left",
        }
        for index in range(number_of_animals)
    ]


def test_grouped_split_never_leaks_animal():
    rows = make_rows()
    # Uma segunda observação por animal torna o teste sensível a split por linha.
    rows.extend(
        {
            **row,
            "image_path": row["image_path"].replace(".jpg", "_visit2.jpg"),
            "event_id": row["event_id"] + "_visit2",
            "weight_kg": str(float(row["weight_kg"]) + 8),
        }
        for row in make_rows()
    )

    result = grouped_split(rows, seed=7, stratify_bins=5)
    assert {row["split"] for row in result} == {"train", "val", "test"}
    assert_no_animal_leakage(result)
    animals_by_split = {
        split: {row["animal_id"] for row in result if row["split"] == split}
        for split in ("train", "val", "test")
    }
    counts = {split: len(animals) for split, animals in animals_by_split.items()}
    assert sum(counts.values()) == 30
    assert counts["train"] >= 20
    assert counts["val"] >= 4
    assert counts["test"] >= 4


def test_grouped_split_is_reproducible():
    first = grouped_split(make_rows(), seed=11)
    second = grouped_split(make_rows(), seed=11)
    assert [row["split"] for row in first] == [row["split"] for row in second]


def test_detects_animal_leakage():
    rows = make_rows(3)
    rows[0]["split"] = "train"
    rows.append(
        {
            **rows[0],
            "image_path": "images/leaked.jpg",
            "event_id": "event_leaked",
            "split": "test",
        }
    )
    with pytest.raises(ValueError, match="Vazamento"):
        assert_no_animal_leakage(rows)


def test_detects_conflicting_weight_in_same_event():
    rows = make_rows(3)
    rows.append(
        {
            **rows[0],
            "image_path": "images/another_view.jpg",
            "weight_kg": "999",
        }
    )
    with pytest.raises(ValueError, match="pesos conflitantes"):
        validate_rows(rows)


def test_manifest_round_trip(tmp_path):
    output = tmp_path / "manifest.csv"
    rows = grouped_split(make_rows(), seed=42)
    write_manifest(rows, output)
    loaded = read_manifest(output)
    assert loaded == rows

    with output.open(encoding="utf-8", newline="") as file:
        assert "split" in next(csv.reader(file))


def test_validates_additional_non_image_file(tmp_path):
    rows = make_rows(3)
    rows[0]["point_cloud_path"] = "clouds/missing.ply"

    with pytest.raises(ValueError, match="arquivo não encontrado"):
        validate_rows(
            rows,
            manifest_path=tmp_path / "manifest.csv",
            image_root=tmp_path,
            check_images=True,
            additional_file_columns=("point_cloud_path",),
        )
