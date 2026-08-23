from pathlib import Path

import pytest
from PIL import Image

from ms_peso.prepare_rgb_cache import prepare_rgb_cache


def create_rows(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    image_root = tmp_path / "data"
    source = image_root / "raw" / "source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (80, 40), color=(10, 20, 30)).save(source)
    rows = [
        {
            "image_path": "raw/source.png",
            "animal_id": "animal_001",
            "event_id": "event_001",
            "weight_kg": "400",
            "view": "side",
            "split": "train",
        }
    ]
    return image_root, rows


def test_prepares_reproducible_rgb_cache(tmp_path: Path) -> None:
    image_root, rows = create_rows(tmp_path)
    output_dir = image_root / "interim" / "cache"
    derived = prepare_rgb_cache(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=image_root,
        output_dir=output_dir,
        image_size=32,
    )
    output_path = image_root / derived[0]["image_path"]
    with Image.open(output_path) as image:
        assert image.size == (32, 32)
    assert derived[0]["source_image_path"] == "raw/source.png"
    assert derived[0]["split"] == "train"
    assert derived[0]["derived_transform"] == "resize_bilinear_32x32"


def test_refuses_to_overwrite_existing_cache(tmp_path: Path) -> None:
    image_root, rows = create_rows(tmp_path)
    output_dir = image_root / "interim" / "cache"
    output_dir.mkdir(parents=True)
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="não será sobrescrito"):
        prepare_rgb_cache(
            rows,
            manifest_path=tmp_path / "manifest.csv",
            image_root=image_root,
            output_dir=output_dir,
            image_size=32,
        )


def test_rejects_cache_outside_image_root(tmp_path: Path) -> None:
    image_root, rows = create_rows(tmp_path)
    with pytest.raises(ValueError, match="deve estar dentro"):
        prepare_rgb_cache(
            rows,
            manifest_path=tmp_path / "manifest.csv",
            image_root=image_root,
            output_dir=tmp_path / "outside",
            image_size=32,
        )
