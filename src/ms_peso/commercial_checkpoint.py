from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ms_peso.commercial_data import CommercialDataContract
from ms_peso.integrity import calculate_sha256


@dataclass(frozen=True)
class CommercialFitCheckpoint:
    """Checkpoint de ajuste comercial com identidade e proveniência verificadas."""

    path: Path
    sha256: str
    architecture: str
    dropout: float
    target_mean: float
    target_std: float
    image_size: int
    input_view: str
    state_dict: Mapping[str, Any]


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuração {key} ausente ou inválida.")
    return value


def load_commercial_fit_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    data_contract: CommercialDataContract,
    image_size: int,
    input_view: str,
) -> CommercialFitCheckpoint:
    """Carrega apenas um checkpoint aleatório ligado ao snapshot informado."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint de ajuste não encontrado: {path}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256.lower()
    ):
        raise ValueError("checkpoint_sha256 deve conter um SHA-256 válido.")
    actual_sha256 = calculate_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("Hash do checkpoint diverge da configuração.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint de ajuste inválido.")
    required_fields = {
        "model_state_dict",
        "architecture",
        "dropout",
        "target_mean",
        "target_std",
        "config",
        "workflow",
        "initialization",
        "commercial_use_allowed",
        "promotion_status",
        "held_out_partitions",
        "source_snapshot_id",
        "source_manifest_sha256",
        "source_split_report_sha256",
    }
    missing = required_fields - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint de ajuste sem campos obrigatórios: "
            + ", ".join(sorted(missing))
        )
    if checkpoint["workflow"] != "commercial_fit":
        raise ValueError("A trilha comercial aceita somente checkpoint de ajuste.")
    if checkpoint["initialization"] != "random":
        raise ValueError("O checkpoint comercial não possui inicialização aleatória.")
    if checkpoint["commercial_use_allowed"] is not False:
        raise ValueError(
            "O checkpoint de ajuste possui autorização comercial indevida."
        )
    if checkpoint["promotion_status"] != "not_promoted":
        raise ValueError("O checkpoint de ajuste já possui estado de promoção.")
    held_out = checkpoint["held_out_partitions"]
    if not isinstance(held_out, list) or not {"calibration", "test"}.issubset(
        set(held_out)
    ):
        raise ValueError("Checkpoint sem partições reservadas de calibração e teste.")

    provenance_checks = {
        "source_snapshot_id": data_contract.snapshot_id,
        "source_manifest_sha256": data_contract.manifest_sha256,
        "source_split_report_sha256": data_contract.split_report_sha256,
    }
    for key, expected in provenance_checks.items():
        if checkpoint[key] != expected:
            raise ValueError(f"{key} diverge entre checkpoint e dados comerciais.")

    fit_config = checkpoint["config"]
    if not isinstance(fit_config, dict):
        raise ValueError("Checkpoint sem configuração de ajuste válida.")
    fit_project = _required_mapping(fit_config, "project")
    fit_model = _required_mapping(fit_config, "model")
    fit_data = _required_mapping(fit_config, "data")
    if fit_project.get("workflow") != "commercial_fit":
        raise ValueError("Configuração interna não pertence a commercial_fit.")
    if fit_model.get("pretrained") is not False:
        raise ValueError("Configuração interna contém pesos pré-treinados.")
    if fit_model.get("initialization") != "random":
        raise ValueError("Configuração interna não declara inicialização aleatória.")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size deve ser um inteiro positivo.")
    if image_size != fit_data.get("image_size"):
        raise ValueError("Tamanho de imagem diverge do ajuste comercial.")
    if not isinstance(input_view, str) or not input_view.strip():
        raise ValueError("input_view deve identificar a vista esperada.")
    if input_view != fit_data.get("view"):
        raise ValueError("Vista de entrada diverge do ajuste comercial.")

    architecture = checkpoint["architecture"]
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("Arquitetura do checkpoint é inválida.")
    try:
        dropout = float(checkpoint["dropout"])
        target_mean = float(checkpoint["target_mean"])
        target_std = float(checkpoint["target_std"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadados numéricos do checkpoint são inválidos.") from exc
    if not 0 <= dropout <= 1:
        raise ValueError("Dropout do checkpoint é inválido.")
    if not math.isfinite(target_mean) or not math.isfinite(target_std):
        raise ValueError("Normalização do checkpoint contém valor não finito.")
    if target_std <= 0:
        raise ValueError("Desvio-padrão do checkpoint deve ser positivo.")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("Estado do modelo no checkpoint é inválido.")

    return CommercialFitCheckpoint(
        path=path,
        sha256=actual_sha256,
        architecture=architecture,
        dropout=dropout,
        target_mean=target_mean,
        target_std=target_std,
        image_size=image_size,
        input_view=input_view,
        state_dict=state_dict,
    )
