from pathlib import Path

import torch
from PIL import Image

from ms_peso.dataset import RGBDepthCattleWeightDataset


def test_loads_rgb_and_depth_as_four_channels(tmp_path: Path) -> None:
    Image.new("RGB", (16, 12), color=(100, 120, 140)).save(tmp_path / "rgb.png")
    Image.new("I;16", (8, 6), color=4000).save(tmp_path / "depth.png")
    rows = [
        {
            "image_path": "rgb.png",
            "depth_image_path": "depth.png",
            "animal_id": "cow_001",
            "event_id": "event_001",
            "weight_kg": "450",
        }
    ]
    dataset = RGBDepthCattleWeightDataset(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        depth_image_column="depth_image_path",
        image_size=32,
        training=False,
        depth_max_mm=8000,
        target_mean=400,
        target_std=50,
    )

    tensor, target, index = dataset[0]

    assert tensor.shape == (4, 32, 32)
    assert torch.allclose(tensor[3], torch.zeros(32, 32))
    assert target == 1.0
    assert index == 0
