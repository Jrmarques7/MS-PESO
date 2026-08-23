from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from ms_peso.dataset import build_transform
from ms_peso.model import build_model
from ms_peso.model_package import ModelDescriptor, verify_model_package

SUPPORTED_RGB_ARCHITECTURES = {"resnet18", "efficientnet_b0", "convnext_tiny"}


@dataclass(frozen=True)
class WeightPrediction:
    estimated_weight_kg: float
    original_width: int
    original_height: int


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("Dispositivo deve ser auto, cpu ou cuda.")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA foi solicitada, mas não está disponível.")
    return torch.device(normalized)


class CattleWeightPredictor:
    """Executa somente pré-processamento e inferência de uma imagem RGB."""

    def __init__(
        self,
        *,
        descriptor: ModelDescriptor,
        model: nn.Module,
        device: torch.device,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.descriptor = descriptor
        self.model = model
        self.device = device
        self.target_mean = target_mean
        self.target_std = target_std
        self.transform = build_transform(descriptor.image_size, training=False)

    @classmethod
    def load(
        cls, descriptor: ModelDescriptor, *, device: str = "auto"
    ) -> CattleWeightPredictor:
        verify_model_package(descriptor)
        if descriptor.architecture not in SUPPORTED_RGB_ARCHITECTURES:
            raise ValueError(
                "A inferência de imagem única aceita somente arquiteturas RGB."
            )
        checkpoint = torch.load(
            descriptor.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        required = {
            "model_state_dict",
            "architecture",
            "dropout",
            "target_mean",
            "target_std",
            "config",
        }
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(
                "Checkpoint sem campos obrigatórios: " + ", ".join(sorted(missing))
            )
        if checkpoint["architecture"] != descriptor.architecture:
            raise ValueError("Arquitetura diverge entre checkpoint e descritor.")
        checkpoint_image_size = checkpoint["config"]["data"]["image_size"]
        if int(checkpoint_image_size) != descriptor.image_size:
            raise ValueError("Tamanho de imagem diverge entre checkpoint e descritor.")

        target_mean = float(checkpoint["target_mean"])
        target_std = float(checkpoint["target_std"])
        if not math.isfinite(target_mean) or not math.isfinite(target_std):
            raise ValueError("Normalização do alvo contém valores não finitos.")
        if target_std <= 0:
            raise ValueError("Desvio-padrão do alvo deve ser positivo.")
        dropout = float(checkpoint["dropout"])
        if not 0 <= dropout <= 1:
            raise ValueError("Dropout do checkpoint é inválido.")

        selected_device = resolve_device(device)
        model = build_model(
            descriptor.architecture,
            pretrained=False,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(selected_device)
        model.eval()
        return cls(
            descriptor=descriptor,
            model=model,
            device=selected_device,
            target_mean=target_mean,
            target_std=target_std,
        )

    def predict_image(self, path: str | Path) -> WeightPrediction:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        try:
            with Image.open(image_path) as image:
                original_width, original_height = image.size
                tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Arquivo de imagem inválido: {image_path}") from exc

        with torch.inference_mode():
            normalized_prediction = self.model(tensor.to(self.device)).reshape(-1)
        if normalized_prediction.numel() != 1:
            raise RuntimeError("O modelo não retornou exatamente uma predição.")
        estimated_weight = (
            float(normalized_prediction.item()) * self.target_std + self.target_mean
        )
        if not math.isfinite(estimated_weight):
            raise RuntimeError("O modelo produziu uma estimativa não finita.")
        return WeightPrediction(
            estimated_weight_kg=estimated_weight,
            original_width=original_width,
            original_height=original_height,
        )
