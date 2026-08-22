from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ms_peso.manifest import resolve_image_path


def build_transform(image_size: int, training: bool):
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class CattleWeightDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        manifest_path: str | Path,
        image_root: str | Path | None,
        image_size: int,
        training: bool,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.rows = rows
        self.manifest_path = Path(manifest_path)
        self.image_root = image_root
        self.transform = build_transform(image_size, training)
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_image_path(row, self.manifest_path, self.image_root)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        weight = float(row["weight_kg"])
        normalized_weight = (weight - self.target_mean) / self.target_std
        return tensor, normalized_weight, index
