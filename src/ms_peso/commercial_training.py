from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import (
    COMMERCIAL_SPLITS,
    assert_no_animal_leakage,
    validate_rows,
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


@dataclass(frozen=True)
class CommercialFitContract:
    """Identidade aprovada da entrada de um ajuste comercial ainda não promovido."""

    snapshot_id: str
    manifest_sha256: str
    split_report_sha256: str
    number_of_animals: dict[str, int]
    number_of_images: dict[str, int]


def _load_split_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Relatório de split não encontrado: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Relatório de split não contém JSON válido.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Relatório de split inválido.")
    return payload


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

    validate_rows(rows)
    if not all(row.get("split") for row in rows):
        raise ValueError(
            "O ajuste comercial não cria splits automaticamente; use um manifesto "
            "comercial aprovado."
        )
    actual_splits = {row["split"] for row in rows}
    if actual_splits != set(COMMERCIAL_SPLITS):
        raise ValueError(
            "O manifesto comercial deve conter exatamente train, val, calibration "
            "e test."
        )
    assert_no_animal_leakage(rows)

    report = _load_split_report(Path(split_report_path))
    if report.get("status") != "passed":
        raise ValueError("O manifesto não possui um relatório de split aprovado.")
    snapshot_id = report.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("Relatório de split sem snapshot_id válido.")

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Relatório de split sem proveniência.")
    manifest_sha256 = calculate_sha256(manifest_path)
    if provenance.get("output_manifest_sha256") != manifest_sha256:
        raise ValueError("Hash do manifesto diverge do relatório de split.")

    image_counts = Counter(row["split"] for row in rows)
    animal_counts = {
        split: len({row["animal_id"] for row in rows if row["split"] == split})
        for split in COMMERCIAL_SPLITS
    }
    reported_splits = report.get("splits")
    if not isinstance(reported_splits, dict):
        raise ValueError("Relatório de split sem contagens por partição.")
    for split in COMMERCIAL_SPLITS:
        reported = reported_splits.get(split)
        if not isinstance(reported, dict):
            raise ValueError(f"Relatório de split sem contagens para {split}.")
        if reported.get("images") != image_counts[split]:
            raise ValueError(f"Contagem de imagens de {split} diverge do relatório.")
        if reported.get("animals") != animal_counts[split]:
            raise ValueError(f"Contagem de animais de {split} diverge do relatório.")

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            "A saída do ajuste comercial já existe; escolha uma nova versão."
        )

    return CommercialFitContract(
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        split_report_sha256=calculate_sha256(split_report_path),
        number_of_animals=animal_counts,
        number_of_images={split: image_counts[split] for split in COMMERCIAL_SPLITS},
    )
