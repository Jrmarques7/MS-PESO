from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable
from pathlib import Path

SUPPORTED_ANGLES = tuple(f"angle_{number}" for number in range(1, 6))
SOURCE_DATASET = "mendeley_multiview_weight_v1"
SOURCE_DOI = "10.17632/vf7pxfs7dx.1"

_IMAGE_PATTERN = re.compile(r"^cow_(?P<animal>\d{4})_angle(?P<angle>[1-5])\.jpg$")
_REQUIRED_METADATA_COLUMNS = {
    "image_name",
    "cow_id",
    "angle",
    "collection_date",
    "time_of_day",
    "weather",
    "device",
}


def resolve_multiview_root(dataset_root: str | Path) -> Path:
    """Resolve a pasta Cow_Images sem depender do diretório externo do ZIP."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {root}")
    if (root / "labels.csv").is_file() and (root / "metadata.csv").is_file():
        return root

    candidates = [
        path.parent
        for path in root.rglob("labels.csv")
        if (path.parent / "metadata.csv").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Era esperada exatamente uma pasta com labels.csv e metadata.csv; "
            f"foram encontradas {len(candidates)}."
        )
    return candidates[0]


def read_multiview_labels(dataset_root: str | Path) -> dict[str, str]:
    """Lê e valida a tabela de pesos publicada pelo dataset."""
    root = resolve_multiview_root(dataset_root)
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"cow_id", "weight_kg"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("labels.csv não contém cow_id e weight_kg.")
        labels: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            source_id = _normalize_source_id(row.get("cow_id"), line_number)
            weight = _normalize_weight(row.get("weight_kg"), line_number)
            if source_id in labels:
                raise ValueError(f"Animal duplicado em labels.csv: {source_id}")
            labels[source_id] = weight
    if not labels:
        raise ValueError("labels.csv não contém animais.")
    return labels


def read_multiview_metadata(
    dataset_root: str | Path,
) -> dict[tuple[str, str], dict[str, str]]:
    """Lê metadados úteis sem propagar GPS exato ou nome do coletor."""
    root = resolve_multiview_root(dataset_root)
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing = _REQUIRED_METADATA_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                "metadata.csv não contém as colunas esperadas: "
                + ", ".join(sorted(missing))
            )

        metadata: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            source_id = _normalize_source_id(row.get("cow_id"), line_number)
            angle = _normalize_angle(row.get("angle"), line_number)
            image_name = (row.get("image_name") or "").strip()
            match = _IMAGE_PATTERN.fullmatch(image_name)
            if not match:
                raise ValueError(
                    f"linha {line_number}: nome de imagem inesperado: {image_name!r}"
                )
            if int(match.group("animal")) != int(source_id) or int(
                match.group("angle")
            ) != int(angle.removeprefix("angle_")):
                raise ValueError(
                    f"linha {line_number}: imagem, animal e ângulo não correspondem."
                )
            key = (source_id, angle)
            if key in metadata:
                raise ValueError(f"Metadado duplicado para animal/vista: {key}")
            metadata[key] = {
                "image_name": image_name,
                "collection_date": (row.get("collection_date") or "").strip(),
                "time_of_day": (row.get("time_of_day") or "").strip(),
                "weather": (row.get("weather") or "").strip(),
                "device": (row.get("device") or "").strip(),
                "location": (row.get("location") or "").strip(),
            }
    if not metadata:
        raise ValueError("metadata.csv não contém imagens.")
    return metadata


def build_multiview_manifest_rows(
    dataset_root: str | Path,
    *,
    image_root: str | Path,
    angles: Iterable[str] = ("angle_1",),
    acknowledge_unverified_source_labels: bool = False,
) -> list[dict[str, str]]:
    """Compõe um manifesto de revisão, com aceite explícito dos rótulos."""
    if not acknowledge_unverified_source_labels:
        raise ValueError(
            "Os pesos desta fonte ainda não foram validados independentemente. "
            "Use acknowledge_unverified_source_labels=True somente para gerar "
            "um manifesto de revisão, não um conjunto comercial aprovado."
        )

    selected_angles = _validate_angles(angles)
    root = resolve_multiview_root(dataset_root)
    labels = read_multiview_labels(root)
    metadata = read_multiview_metadata(root)
    image_base = Path(image_root).resolve()

    expected_keys = {
        (source_id, angle) for source_id in labels for angle in SUPPORTED_ANGLES
    }
    metadata_keys = set(metadata)
    if metadata_keys != expected_keys:
        missing = sorted(expected_keys - metadata_keys)[:10]
        extra = sorted(metadata_keys - expected_keys)[:10]
        raise ValueError(
            "A grade animal/ângulo está incompleta ou possui itens extras. "
            f"Ausentes: {missing}; extras: {extra}."
        )

    rows: list[dict[str, str]] = []
    for source_id in sorted(labels, key=int):
        animal_id = f"mendeley_multiview_{int(source_id):04d}"
        for angle in selected_angles:
            item = metadata[(source_id, angle)]
            angle_number = angle.removeprefix("angle_")
            image_path = root / f"Angle{angle_number}" / item["image_name"]
            if not image_path.is_file():
                raise ValueError(
                    f"Imagem ausente para animal {source_id}, vista {angle}: "
                    f"{image_path}"
                )
            relative_path = _relative_to_image_root(image_path, image_base)
            rows.append(
                {
                    "image_path": relative_path.as_posix(),
                    "animal_id": animal_id,
                    "event_id": f"{animal_id}_capture_001",
                    "weight_kg": labels[source_id],
                    "view": angle,
                    "quality": "review_required",
                    "label_status": "unverified_source_label",
                    "training_eligible": "false",
                    "source_dataset": SOURCE_DATASET,
                    "source_version": "1",
                    "source_doi": SOURCE_DOI,
                    "source_license": "CC_BY_4_0",
                    "source_animal_id": source_id,
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "image_name" and value
                    },
                }
            )
    return rows


def _validate_angles(angles: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(angles))
    invalid = set(selected) - set(SUPPORTED_ANGLES)
    if invalid:
        raise ValueError(f"Ângulos inválidos: {sorted(invalid)}")
    if not selected:
        raise ValueError("Selecione ao menos um ângulo.")
    return selected


def _normalize_source_id(value: object, line_number: int) -> str:
    text = str(value or "").strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"linha {line_number}: cow_id inválido: {text!r}")
    return str(int(text))


def _normalize_angle(value: object, line_number: int) -> str:
    text = str(value or "").strip()
    if not text.isdigit() or f"angle_{int(text)}" not in SUPPORTED_ANGLES:
        raise ValueError(f"linha {line_number}: ângulo inválido: {text!r}")
    return f"angle_{int(text)}"


def _normalize_weight(value: object, line_number: int) -> str:
    text = str(value or "").strip()
    try:
        weight = float(text)
    except ValueError as exc:
        raise ValueError(f"linha {line_number}: peso inválido: {text!r}") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"linha {line_number}: peso deve ser positivo e finito.")
    return str(int(weight)) if weight.is_integer() else str(weight)


def _relative_to_image_root(path: Path, image_root: Path) -> Path:
    try:
        return path.resolve().relative_to(image_root)
    except ValueError as exc:
        raise ValueError(
            f"Imagem {path} não está dentro de image_root {image_root}."
        ) from exc
