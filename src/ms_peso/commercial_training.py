from __future__ import annotations

from pathlib import Path
from typing import Any

from ms_peso.commercial_data import (
    CommercialDataContract,
    validate_commercial_data_contract,
)

_FORBIDDEN_WEIGHT_KEYS = (
    "checkpoint",
    "checkpoint_path",
    "initial_checkpoint",
    "initialization_checkpoint",
    "resume_from",
    "teacher_checkpoint",
    "weights",
    "weights_path",
)


CommercialFitContract = CommercialDataContract


def validate_commercial_fit_contract(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    manifest_path: str | Path,
    split_report_path: str | Path,
    output_dir: str | Path,
) -> CommercialFitContract:
    """Valida a proveniência e o isolamento exigidos antes do ajuste comercial."""
    project_config = config.get("project")
    if not isinstance(project_config, dict):
        raise ValueError("Configuração project ausente ou inválida.")
    if project_config.get("workflow") != "commercial_fit":
        raise ValueError("A trilha comercial exige project.workflow: commercial_fit.")

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Configuração model ausente ou inválida.")
    if model_config.get("pretrained") is not False:
        raise ValueError("O ajuste comercial exige model.pretrained: false.")
    if model_config.get("initialization") != "random":
        raise ValueError("O ajuste comercial exige model.initialization: random.")
    forbidden_sources = [
        key for key in _FORBIDDEN_WEIGHT_KEYS if model_config.get(key) not in (None, "")
    ]
    if forbidden_sources:
        raise ValueError(
            "O ajuste comercial não aceita pesos ou checkpoints de origem: "
            + ", ".join(forbidden_sources)
        )

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            "A saída do ajuste comercial já existe; escolha uma nova versão."
        )

    return validate_commercial_data_contract(
        rows,
        manifest_path=manifest_path,
        split_report_path=split_report_path,
    )
