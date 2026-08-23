import csv
from pathlib import Path

from PIL import Image

from ms_peso.prepare_ssl_manifest import (
    build_ssl_manifest_rows,
    read_ssl_manifest,
    write_ssl_manifest,
)


def write_source_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_combines_train_and_unsplit_without_duplicates(tmp_path: Path) -> None:
    image_root = tmp_path / "data"
    image_root.mkdir()
    for name, color in (("a.png", 10), ("b.png", 20), ("copy.png", 10)):
        Image.new("RGB", (16, 16), color=(color, 0, 0)).save(image_root / name)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base = {
        "event_id": "capture",
        "weight_kg": "400",
        "view": "side",
        "source_license": "CC_BY_4_0",
    }
    write_source_manifest(
        first,
        [
            {
                **base,
                "image_path": "a.png",
                "animal_id": "a",
                "split": "train",
                "source_dataset": "horqin",
            },
            {
                **base,
                "image_path": "b.png",
                "animal_id": "b",
                "split": "test",
                "source_dataset": "horqin",
            },
        ],
    )
    write_source_manifest(
        second,
        [
            {
                **base,
                "image_path": "copy.png",
                "animal_id": "c",
                "split": "",
                "source_dataset": "multiview",
            },
            {
                **base,
                "image_path": "b.png",
                "animal_id": "d",
                "split": "",
                "source_dataset": "multiview",
            },
        ],
    )
    rows, stats = build_ssl_manifest_rows(
        [first, second], image_root=image_root
    )
    assert len(rows) == 2
    assert stats == {
        "candidates": 3,
        "retained": 2,
        "duplicates_removed": 1,
        "non_train_rows_skipped": 1,
    }
    assert {row["source_split"] for row in rows} == {"train", "unsplit"}
    output = tmp_path / "ssl.csv"
    write_ssl_manifest(rows, output)
    assert read_ssl_manifest(output) == rows
