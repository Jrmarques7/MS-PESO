from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ms_peso.config import load_yaml_config


@dataclass(frozen=True)
class ImageQualityPolicy:
    policy_id: str
    policy_version: str
    analysis_max_dimension: int
    min_width: int
    min_height: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_mean_luma: float
    max_mean_luma: float
    max_dark_fraction: float
    max_bright_fraction: float
    min_sharpness: float
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class QualityCheck:
    code: str
    passed: bool
    value: object
    requirement: str
    rejection_message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "passed": self.passed,
            "value": self.value,
            "requirement": self.requirement,
        }


@dataclass(frozen=True)
class ImageQualityReport:
    policy_id: str
    policy_version: str
    width: int
    height: int
    checks: tuple[QualityCheck, ...]
    limitations: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def status(self) -> str:
        return "passed" if self.accepted else "rejected"

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(
            check.rejection_message for check in self.checks if not check.passed
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "policy": {
                "id": self.policy_id,
                "version": self.policy_version,
            },
            "checks": [check.to_dict() for check in self.checks],
            "rejection_reasons": list(self.rejection_reasons),
            "limitations": list(self.limitations),
        }


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Política de qualidade sem objeto {key!r}.")
    return value


def _required_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Política de qualidade sem texto {key!r}.")
    return value.strip()


def _required_number(
    config: dict[str, Any], key: str, *, minimum: float | None = None
) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Política de qualidade sem número {key!r}.")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"Valor inválido para {key!r} na política de qualidade.")
    return result


def _required_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Política de qualidade sem inteiro positivo {key!r}.")
    return value


def load_image_quality_policy(path: str | Path) -> ImageQualityPolicy:
    config = load_yaml_config(path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão da política de qualidade não suportada.")
    analysis = _required_mapping(config, "analysis")
    thresholds = _required_mapping(config, "thresholds")
    limitations = config.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("limitations deve ser uma lista de textos não vazia.")

    min_aspect_ratio = _required_number(
        thresholds, "min_aspect_ratio", minimum=0.01
    )
    max_aspect_ratio = _required_number(
        thresholds, "max_aspect_ratio", minimum=0.01
    )
    min_mean_luma = _required_number(thresholds, "min_mean_luma", minimum=0)
    max_mean_luma = _required_number(thresholds, "max_mean_luma", minimum=0)
    max_dark_fraction = _required_number(
        thresholds, "max_dark_fraction", minimum=0
    )
    max_bright_fraction = _required_number(
        thresholds, "max_bright_fraction", minimum=0
    )
    if min_aspect_ratio > max_aspect_ratio:
        raise ValueError("Intervalo de proporção da política é inválido.")
    if min_mean_luma > max_mean_luma or max_mean_luma > 255:
        raise ValueError("Intervalo de luminosidade da política é inválido.")
    if max_dark_fraction > 1 or max_bright_fraction > 1:
        raise ValueError("Frações de exposição devem estar entre zero e um.")

    return ImageQualityPolicy(
        policy_id=_required_text(config, "policy_id"),
        policy_version=_required_text(config, "policy_version"),
        analysis_max_dimension=_required_positive_int(analysis, "max_dimension"),
        min_width=_required_positive_int(thresholds, "min_width"),
        min_height=_required_positive_int(thresholds, "min_height"),
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_mean_luma=min_mean_luma,
        max_mean_luma=max_mean_luma,
        max_dark_fraction=max_dark_fraction,
        max_bright_fraction=max_bright_fraction,
        min_sharpness=_required_number(thresholds, "min_sharpness", minimum=0),
        limitations=tuple(item.strip() for item in limitations),
    )


def _sharpness_score(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    core = gray[1:-1, 1:-1]
    laplacian = (
        -4 * core
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def assess_image_quality(
    path: str | Path, policy: ImageQualityPolicy
) -> ImageQualityReport:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            analysis_image = source.convert("RGB")
            analysis_image.thumbnail(
                (policy.analysis_max_dimension, policy.analysis_max_dimension),
                Image.Resampling.LANCZOS,
            )
            gray = np.asarray(analysis_image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Arquivo de imagem inválido: {image_path}") from exc

    aspect_ratio = width / height
    mean_luma = float(gray.mean())
    dark_fraction = float((gray <= 10).mean())
    bright_fraction = float((gray >= 245).mean())
    sharpness = _sharpness_score(gray)
    checks = (
        QualityCheck(
            code="minimum_resolution",
            passed=width >= policy.min_width and height >= policy.min_height,
            value={"width": width, "height": height},
            requirement=f">= {policy.min_width}x{policy.min_height}",
            rejection_message=(
                "Resolução insuficiente; aproxime a câmera ou use uma imagem maior."
            ),
        ),
        QualityCheck(
            code="landscape_aspect_ratio",
            passed=policy.min_aspect_ratio <= aspect_ratio <= policy.max_aspect_ratio,
            value=round(aspect_ratio, 4),
            requirement=(
                f"entre {policy.min_aspect_ratio:.2f} e "
                f"{policy.max_aspect_ratio:.2f}"
            ),
            rejection_message="Use uma foto horizontal, sem corte excessivo.",
        ),
        QualityCheck(
            code="mean_luma",
            passed=policy.min_mean_luma <= mean_luma <= policy.max_mean_luma,
            value=round(mean_luma, 4),
            requirement=(
                f"entre {policy.min_mean_luma:.1f} e {policy.max_mean_luma:.1f}"
            ),
            rejection_message="A foto está escura ou clara demais; ajuste a exposição.",
        ),
        QualityCheck(
            code="dark_clipping",
            passed=dark_fraction <= policy.max_dark_fraction,
            value=round(dark_fraction, 6),
            requirement=f"<= {policy.max_dark_fraction:.2f}",
            rejection_message="Há pixels escuros demais em grande parte da foto.",
        ),
        QualityCheck(
            code="bright_clipping",
            passed=bright_fraction <= policy.max_bright_fraction,
            value=round(bright_fraction, 6),
            requirement=f"<= {policy.max_bright_fraction:.2f}",
            rejection_message="Há pixels estourados em grande parte da foto.",
        ),
        QualityCheck(
            code="sharpness",
            passed=sharpness >= policy.min_sharpness,
            value=round(sharpness, 4),
            requirement=f">= {policy.min_sharpness:.1f}",
            rejection_message="A foto parece borrada; estabilize a câmera e refaça.",
        ),
    )
    return ImageQualityReport(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        width=width,
        height=height,
        checks=checks,
        limitations=policy.limitations,
    )
