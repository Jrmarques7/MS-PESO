from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import COMMERCIAL_SPLITS, validate_rows


@dataclass(frozen=True)
class CommercialDataContract:
    """Identidade verificada de um snapshot com quatro partições comerciais."""

    snapshot_id: str
    manifest_sha256: str
    split_report_sha256: str
    number_of_animals: dict[str, int]
    number_of_images: dict[str, int]


def _load_split_report(path: Path) -> dict[str, object]:
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


def validate_commercial_data_contract(
    rows: list[dict[str, str]],
    *,
    manifest_path: str | Path,
    split_report_path: str | Path,
) -> CommercialDataContract:
    """Confere split, contagens e hashes sem abrir qualquer imagem."""
    validate_rows(rows)
    if not all(row.get("split") for row in rows):
        raise ValueError(
            "A trilha comercial não cria splits automaticamente; use um manifesto "
            "comercial aprovado."
        )
    actual_splits = {row["split"] for row in rows}
    if actual_splits != set(COMMERCIAL_SPLITS):
        raise ValueError(
            "O manifesto comercial deve conter exatamente train, val, calibration "
            "e test."
        )

    report_path = Path(split_report_path)
    report = _load_split_report(report_path)
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

    return CommercialDataContract(
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        split_report_sha256=calculate_sha256(report_path),
        number_of_animals=animal_counts,
        number_of_images={split: image_counts[split] for split in COMMERCIAL_SPLITS},
    )
