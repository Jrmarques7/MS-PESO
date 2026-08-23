from pathlib import Path

import pytest
import torch
from PIL import Image

from ms_peso.dataset import MultiViewCattleWeightDataset
from ms_peso.multi_view import pair_views


def make_rows() -> list[dict[str, str]]:
    rows = []
    for view in ("left", "top"):
        rows.append(
            {
                "image_path": f"images/cow_001_{view}.png",
                "animal_id": "cow_001",
                "event_id": "event_001",
                "weight_kg": "420",
                "view": view,
                "split": "train",
            }
        )
    return rows


def test_pairs_two_views_into_one_event() -> None:
    paired_rows = pair_views(make_rows(), primary_view="left", secondary_view="top")

    assert len(paired_rows) == 1
    assert paired_rows[0]["image_path"].endswith("left.png")
    assert paired_rows[0]["secondary_image_path"].endswith("top.png")
    assert paired_rows[0]["secondary_view"] == "top"


def test_rejects_missing_or_duplicate_views() -> None:
    with pytest.raises(ValueError, match="número de vistas inválido"):
        pair_views(make_rows()[:1], primary_view="left", secondary_view="top")

    duplicate_rows = [*make_rows(), {**make_rows()[1], "image_path": "another.png"}]
    with pytest.raises(ValueError, match="número de vistas inválido"):
        pair_views(duplicate_rows, primary_view="left", secondary_view="top")


def test_rejects_conflicting_pair_metadata() -> None:
    rows = make_rows()
    rows[1]["weight_kg"] = "421"
    with pytest.raises(ValueError, match="weight_kg conflitante"):
        pair_views(rows, primary_view="left", secondary_view="top")


def test_dataset_returns_two_rgb_views(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (12, 8), color=(255, 0, 0)).save(images / "left.png")
    Image.new("RGB", (12, 8), color=(0, 0, 255)).save(images / "top.png")
    rows = [
        {
            "image_path": "images/left.png",
            "secondary_image_path": "images/top.png",
            "animal_id": "cow_001",
            "event_id": "event_001",
            "weight_kg": "420",
        }
    ]
    dataset = MultiViewCattleWeightDataset(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=None,
        secondary_image_column="secondary_image_path",
        image_size=16,
        training=False,
        target_mean=400,
        target_std=20,
    )

    tensor, target, index = dataset[0]

    assert tensor.shape == (6, 16, 16)
    assert torch.isfinite(tensor).all()
    assert target == 1
    assert index == 0
