from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from ms_peso.commercial_model_package import (
    CommercialCandidateDescriptor,
    verify_commercial_candidate_package,
)
from ms_peso.dataset import build_transform
from ms_peso.inference import SUPPORTED_RGB_ARCHITECTURES, resolve_device
from ms_peso.model import build_model


@dataclass(frozen=True)
class CommercialWeightPrediction:
    estimated_weight_kg: float
    interval_lower_kg: float
    interval_upper_kg: float
    interval_radius_kg: float
    target_coverage: float
    lower_bound_clipped_at_zero: bool
    original_width: int
    original_height: int


class CommercialCandidatePredictor:
    """Estima peso e intervalo de um candidato interno ainda não autorizado."""

    def __init__(
        self,
        *,
        descriptor: CommercialCandidateDescriptor,
        model: nn.Module,
        device: torch.device,
        target_mean: float,
        target_std: float,
        target_coverage: float,
        interval_radius_kg: float,
    ) -> None:
        self.descriptor = descriptor
        self.model = model
        self.device = device
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_coverage = target_coverage
        self.interval_radius_kg = interval_radius_kg
        self.transform = build_transform(descriptor.image_size, training=False)

    @classmethod
    def load(
        cls,
        descriptor: CommercialCandidateDescriptor,
        *,
        device: str = "auto",
    ) -> CommercialCandidatePredictor:
        verified = verify_commercial_candidate_package(descriptor)
        if descriptor.architecture not in SUPPORTED_RGB_ARCHITECTURES:
            raise ValueError(
                "A inferência comercial aceita somente arquiteturas RGB."
            )
        checkpoint = verified.checkpoint
        try:
            dropout = float(checkpoint["dropout"])
            target_mean = float(checkpoint["target_mean"])
            target_std = float(checkpoint["target_std"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Metadados numéricos do checkpoint são inválidos."
            ) from exc
        if not 0 <= dropout <= 1:
            raise ValueError("Dropout do checkpoint é inválido.")
        if not math.isfinite(target_mean) or not math.isfinite(target_std):
            raise ValueError("Normalização do alvo contém valores não finitos.")
        if target_std <= 0:
            raise ValueError("Desvio-padrão do alvo deve ser positivo.")

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
            target_coverage=verified.target_coverage,
            interval_radius_kg=verified.interval_radius_kg,
        )

    def predict_image(self, path: str | Path) -> CommercialWeightPrediction:
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
        raw_lower = estimated_weight - self.interval_radius_kg
        return CommercialWeightPrediction(
            estimated_weight_kg=estimated_weight,
            interval_lower_kg=max(0.0, raw_lower),
            interval_upper_kg=estimated_weight + self.interval_radius_kg,
            interval_radius_kg=self.interval_radius_kg,
            target_coverage=self.target_coverage,
            lower_bound_clipped_at_zero=raw_lower < 0,
            original_width=original_width,
            original_height=original_height,
        )
