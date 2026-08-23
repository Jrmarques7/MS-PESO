from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_peso.config import load_yaml_config


@dataclass(frozen=True)
class VideoInferencePolicy:
    policy_id: str
    policy_version: str
    status: str
    max_duration_seconds: float
    sample_count: int
    max_frame_pixels: int
    min_valid_frames: int
    top_k: int
    min_temporal_gap_seconds: float
    max_frame_spread_kg: float | None
    limitations: tuple[str, ...]


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Política de vídeo sem objeto {key!r}.")
    return value


def _text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Política de vídeo sem texto {key!r}.")
    return value.strip()


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Política de vídeo sem inteiro positivo {key!r}.")
    return value


def _nonnegative_number(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Valor inválido para {key!r} na política de vídeo.")
    return float(value)


def load_video_inference_policy(path: str | Path) -> VideoInferencePolicy:
    config = load_yaml_config(path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão da política de vídeo não suportada.")

    extraction = _mapping(config, "extraction")
    selection = _mapping(config, "selection")
    consensus = _mapping(config, "consensus")
    sample_count = _positive_int(extraction, "sample_count")
    min_valid_frames = _positive_int(selection, "min_valid_frames")
    top_k = _positive_int(selection, "top_k")
    if not min_valid_frames <= top_k <= sample_count:
        raise ValueError(
            "A política de vídeo exige min_valid_frames <= top_k <= sample_count."
        )

    raw_spread = consensus.get("max_frame_spread_kg")
    if raw_spread is None:
        max_frame_spread_kg = None
    else:
        max_frame_spread_kg = _nonnegative_number(
            consensus, "max_frame_spread_kg"
        )
        if max_frame_spread_kg == 0:
            raise ValueError("max_frame_spread_kg deve ser positivo ou nulo.")

    limitations = config.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("limitations deve ser uma lista de textos não vazia.")

    max_duration_seconds = _nonnegative_number(
        extraction, "max_duration_seconds"
    )
    if max_duration_seconds == 0:
        raise ValueError("max_duration_seconds deve ser positivo.")

    return VideoInferencePolicy(
        policy_id=_text(config, "policy_id"),
        policy_version=_text(config, "policy_version"),
        status=_text(config, "status"),
        max_duration_seconds=max_duration_seconds,
        sample_count=sample_count,
        max_frame_pixels=_positive_int(extraction, "max_frame_pixels"),
        min_valid_frames=min_valid_frames,
        top_k=top_k,
        min_temporal_gap_seconds=_nonnegative_number(
            selection, "min_temporal_gap_seconds"
        ),
        max_frame_spread_kg=max_frame_spread_kg,
        limitations=tuple(item.strip() for item in limitations),
    )
