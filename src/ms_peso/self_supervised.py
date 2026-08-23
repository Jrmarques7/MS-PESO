from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0

from ms_peso.integrity import calculate_sha256


@dataclass(frozen=True)
class SelfSupervisedEncoderCheckpoint:
    path: Path
    sha256: str
    architecture: str
    state_dict: Mapping[str, Any]
    source_manifest_sha256: str


def build_contrastive_transform(image_size: int):
    if image_size <= 0:
        raise ValueError("image_size deve ser positivo.")
    color_jitter = transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class ContrastiveImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        image_root: str | Path,
        image_size: int,
    ) -> None:
        self.rows = rows
        self.image_root = Path(image_root)
        self.transform = build_contrastive_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path = self.image_root / self.rows[index]["image_path"]
        with Image.open(path) as image:
            converted = image.convert("RGB")
            first = self.transform(converted)
            second = self.transform(converted)
        return first, second


class ContrastiveEfficientNet(nn.Module):
    def __init__(self, *, projection_dim: int = 128) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim deve ser positivo.")
        self.encoder = efficientnet_b0(weights=None)
        feature_dim = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return functional.normalize(self.projector(features), dim=1)


def nt_xent_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Os dois lotes contrastivos devem ter a mesma matriz N×D.")
    if first.shape[0] < 2:
        raise ValueError("NT-Xent exige ao menos duas imagens por lote.")
    if temperature <= 0:
        raise ValueError("temperature deve ser positiva.")

    embeddings = functional.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits.fill_diagonal_(float("-inf"))
    batch_size = first.shape[0]
    targets = torch.arange(batch_size * 2, device=logits.device)
    targets = (targets + batch_size) % (batch_size * 2)
    return functional.cross_entropy(logits, targets)


def load_self_supervised_encoder(
    path: str | Path,
    *,
    expected_sha256: str,
) -> SelfSupervisedEncoderCheckpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder SSL não encontrado: {checkpoint_path}")
    actual_sha256 = calculate_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("SHA-256 do encoder SSL divergiu.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "model_state_dict",
        "checkpoint_type",
        "architecture",
        "pretrained",
        "initialization",
        "labels_used",
        "commercial_use_allowed",
        "promotion_status",
        "source_license",
        "source_manifest_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "Encoder SSL sem metadados obrigatórios: "
            + ", ".join(sorted(missing))
        )
    expected_values = {
        "checkpoint_type": "self_supervised_encoder",
        "pretrained": False,
        "initialization": "random",
        "labels_used": False,
        "commercial_use_allowed": False,
        "promotion_status": "not_promoted",
        "source_license": "CC_BY_4_0",
    }
    for key, expected in expected_values.items():
        if checkpoint[key] != expected:
            raise ValueError(f"Metadado SSL inválido em {key}.")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("Estado do encoder SSL inválido.")
    return SelfSupervisedEncoderCheckpoint(
        path=checkpoint_path,
        sha256=actual_sha256,
        architecture=str(checkpoint["architecture"]),
        state_dict=state_dict,
        source_manifest_sha256=str(checkpoint["source_manifest_sha256"]),
    )
