from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ms_peso.commercial_data import CommercialDataContract
from ms_peso.conformal import validate_target_coverage
from ms_peso.integrity import calculate_sha256


@dataclass(frozen=True)
class CommercialCalibrationContract:
    """Entrada verificada e autorizada somente para calibração."""

    path: Path
    sha256: str
    architecture: str
    dropout: float
    target_mean: float
    target_std: float
    image_size: int
    input_view: str
    target_coverage: float
    state_dict: Mapping[str, Any]


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuração {key} ausente ou inválida.")
    return value


def validate_commercial_calibration_contract(
    config: dict[str, Any],
    data_contract: CommercialDataContract,
    *,
    output_dir: str | Path,
) -> CommercialCalibrationContract:
    """Valida que a calibração parte do ajuste e do snapshot esperados."""
    project_config = _required_mapping(config, "project")
    if project_config.get("workflow") != "commercial_calibration":
        raise ValueError(
            "A calibração exige project.workflow: commercial_calibration."
        )
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            "A saída da calibração comercial já existe; escolha uma nova versão."
        )

    model_config = _required_mapping(config, "model")
    checkpoint_value = model_config.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError("A calibração exige model.checkpoint.")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint de ajuste não encontrado: {checkpoint_path}"
        )
    expected_sha256 = model_config.get("checkpoint_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256.lower()
    ):
        raise ValueError("model.checkpoint_sha256 deve conter um SHA-256 válido.")
    actual_sha256 = calculate_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("Hash do checkpoint diverge da configuração de calibração.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        raise ValueError("A calibração aceita somente checkpoint de commercial_fit.")
    if checkpoint["initialization"] != "random":
        raise ValueError("O checkpoint comercial não possui inicialização aleatória.")
    if checkpoint["commercial_use_allowed"] is not False:
        raise ValueError(
            "O checkpoint de ajuste possui autorização comercial indevida."
        )
    if checkpoint["promotion_status"] != "not_promoted":
        raise ValueError("A calibração exige um checkpoint ainda não promovido.")
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
            raise ValueError(f"{key} diverge entre checkpoint e dados de calibração.")

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

    data_config = _required_mapping(config, "data")
    image_size = data_config.get("image_size")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("data.image_size deve ser um inteiro positivo.")
    if image_size != fit_data.get("image_size"):
        raise ValueError("Tamanho de imagem diverge do ajuste comercial.")
    input_view = data_config.get("view")
    if not isinstance(input_view, str) or not input_view.strip():
        raise ValueError("data.view deve identificar a vista calibrada.")
    if input_view != fit_data.get("view"):
        raise ValueError("Vista de entrada diverge do ajuste comercial.")

    calibration_config = _required_mapping(config, "calibration")
    try:
        target_coverage = float(calibration_config["target_coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration.target_coverage deve ser numérico.") from exc
    if isinstance(calibration_config.get("target_coverage"), bool):
        raise ValueError("calibration.target_coverage não pode ser booleano.")
    validate_target_coverage(target_coverage)

    architecture = checkpoint["architecture"]
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("Arquitetura do checkpoint é inválida.")
    dropout = float(checkpoint["dropout"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    if not 0 <= dropout <= 1:
        raise ValueError("Dropout do checkpoint é inválido.")
    if not math.isfinite(target_mean) or not math.isfinite(target_std):
        raise ValueError("Normalização do checkpoint contém valor não finito.")
    if target_std <= 0:
        raise ValueError("Desvio-padrão do checkpoint deve ser positivo.")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("Estado do modelo no checkpoint é inválido.")

    return CommercialCalibrationContract(
        path=checkpoint_path,
        sha256=actual_sha256,
        architecture=architecture,
        dropout=dropout,
        target_mean=target_mean,
        target_std=target_std,
        image_size=image_size,
        input_view=input_view,
        target_coverage=target_coverage,
        state_dict=state_dict,
    )
