from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_peso.config import load_yaml_config
from ms_peso.integrity import calculate_sha256

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    model_version: str
    status: str
    production_ready: bool
    commercial_use_allowed: bool
    commercial_blockers: tuple[str, ...]
    architecture: str
    checkpoint_path: Path
    checkpoint_sha256: str
    image_size: int
    input_view: str
    dataset: str
    breed: str
    limitations: tuple[str, ...]
    model_card_path: Path


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Descritor do modelo sem objeto {key!r}.")
    return value


def _required_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Descritor do modelo sem texto {key!r}.")
    return value.strip()


def load_model_descriptor(path: str | Path) -> ModelDescriptor:
    descriptor_path = Path(path).resolve()
    config = load_yaml_config(descriptor_path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão do descritor de modelo não suportada.")

    checkpoint = _required_mapping(config, "checkpoint")
    input_config = _required_mapping(config, "input")
    domain = _required_mapping(config, "domain")
    checkpoint_sha256 = _required_text(checkpoint, "sha256").lower()
    if not SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise ValueError("SHA-256 do checkpoint é inválido.")
    checkpoint_path = (
        descriptor_path.parent / _required_text(checkpoint, "path")
    ).resolve()

    image_size = input_config.get("image_size")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("input.image_size deve ser um inteiro positivo.")
    production_ready = config.get("production_ready")
    if not isinstance(production_ready, bool):
        raise ValueError("production_ready deve ser booleano.")
    commercial_use_allowed = config.get("commercial_use_allowed")
    if not isinstance(commercial_use_allowed, bool):
        raise ValueError("commercial_use_allowed deve ser booleano.")
    commercial_blockers = config.get("commercial_blockers")
    if (
        not isinstance(commercial_blockers, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in commercial_blockers
        )
        or (not commercial_use_allowed and not commercial_blockers)
    ):
        raise ValueError(
            "commercial_blockers deve explicar todo bloqueio de uso comercial."
        )
    limitations = config.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("limitations deve ser uma lista de textos não vazia.")
    model_card_path = (
        descriptor_path.parent / _required_text(config, "model_card")
    ).resolve()

    return ModelDescriptor(
        model_id=_required_text(config, "model_id"),
        model_version=_required_text(config, "model_version"),
        status=_required_text(config, "status"),
        production_ready=production_ready,
        commercial_use_allowed=commercial_use_allowed,
        commercial_blockers=tuple(item.strip() for item in commercial_blockers),
        architecture=_required_text(config, "architecture"),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        image_size=image_size,
        input_view=_required_text(input_config, "view"),
        dataset=_required_text(domain, "dataset"),
        breed=_required_text(domain, "breed"),
        limitations=tuple(item.strip() for item in limitations),
        model_card_path=model_card_path,
    )


def verify_model_package(descriptor: ModelDescriptor) -> None:
    if not descriptor.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint do modelo não encontrado: {descriptor.checkpoint_path}"
        )
    actual_sha256 = calculate_sha256(descriptor.checkpoint_path)
    if actual_sha256 != descriptor.checkpoint_sha256:
        raise ValueError(
            "SHA-256 do checkpoint não corresponde ao descritor: "
            f"esperado {descriptor.checkpoint_sha256}, obtido {actual_sha256}."
        )
    if not descriptor.model_card_path.is_file():
        raise FileNotFoundError(
            f"Model card não encontrado: {descriptor.model_card_path}"
        )
