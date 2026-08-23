from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from ms_peso.manifest import resolve_image_path, resolve_manifest_path


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


class RGBDepthTransform:
    """Aplica geometria sincronizada e normalização específica por modalidade."""

    def __init__(self, image_size: int, training: bool, depth_max_mm: float) -> None:
        if depth_max_mm <= 0:
            raise ValueError("depth_max_mm deve ser positivo.")
        self.image_size = image_size
        self.training = training
        self.depth_max_mm = depth_max_mm
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __call__(
        self, rgb_image: Image.Image, depth_image: Image.Image
    ) -> torch.Tensor:
        output_size = [self.image_size, self.image_size]
        rgb_image = transform_functional.resize(
            rgb_image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        depth_image = transform_functional.resize(
            depth_image,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )
        if self.training and bool(torch.rand(()) < 0.5):
            rgb_image = transform_functional.hflip(rgb_image)
            depth_image = transform_functional.hflip(depth_image)
        if self.training:
            rgb_image = self.color_jitter(rgb_image)

        rgb_tensor = transform_functional.to_tensor(rgb_image)
        rgb_tensor = transform_functional.normalize(
            rgb_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        depth_tensor = transform_functional.pil_to_tensor(depth_image).to(torch.float32)
        depth_tensor = depth_tensor.clamp(0, self.depth_max_mm) / self.depth_max_mm
        depth_tensor = (depth_tensor - 0.5) / 0.5
        return torch.cat((rgb_tensor, depth_tensor), dim=0)


class RGBDepthCattleWeightDataset(Dataset):
    """Carrega exclusivamente um par RGB/profundidade e seu peso."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        manifest_path: str | Path,
        image_root: str | Path | None,
        depth_image_column: str,
        image_size: int,
        training: bool,
        depth_max_mm: float,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.rows = rows
        self.manifest_path = Path(manifest_path)
        self.image_root = image_root
        self.depth_image_column = depth_image_column
        self.transform = RGBDepthTransform(image_size, training, depth_max_mm)
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        rgb_path = resolve_image_path(row, self.manifest_path, self.image_root)
        depth_path = resolve_manifest_path(
            row,
            self.depth_image_column,
            self.manifest_path,
            self.image_root,
        )
        with Image.open(rgb_path) as rgb_image, Image.open(depth_path) as depth_image:
            tensor = self.transform(rgb_image.convert("RGB"), depth_image)
        weight = float(row["weight_kg"])
        normalized_weight = (weight - self.target_mean) / self.target_std
        return tensor, normalized_weight, index
