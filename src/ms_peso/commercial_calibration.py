from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_peso.commercial_checkpoint import (
    CommercialFitCheckpoint,
    load_commercial_fit_checkpoint,
)
from ms_peso.commercial_data import CommercialDataContract
from ms_peso.conformal import validate_target_coverage


@dataclass(frozen=True)
class CommercialCalibrationContract:
    """Entrada verificada e autorizada somente para calibração."""

    checkpoint: CommercialFitCheckpoint
    target_coverage: float


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
    expected_sha256 = model_config.get("checkpoint_sha256")
    data_config = _required_mapping(config, "data")
    image_size = data_config.get("image_size")
    input_view = data_config.get("view")
    checkpoint = load_commercial_fit_checkpoint(
        checkpoint_value,
        expected_sha256=expected_sha256,
        data_contract=data_contract,
        image_size=image_size,
        input_view=input_view,
    )

    calibration_config = _required_mapping(config, "calibration")
    try:
        target_coverage = float(calibration_config["target_coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration.target_coverage deve ser numérico.") from exc
    if isinstance(calibration_config.get("target_coverage"), bool):
        raise ValueError("calibration.target_coverage não pode ser booleano.")
    validate_target_coverage(target_coverage)

    return CommercialCalibrationContract(
        checkpoint=checkpoint,
        target_coverage=target_coverage,
    )
