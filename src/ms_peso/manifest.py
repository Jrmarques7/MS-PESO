from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import median

REQUIRED_COLUMNS = ("image_path", "animal_id", "event_id", "weight_kg")
BASELINE_SPLITS = ("train", "val", "test")
COMMERCIAL_SPLITS = ("train", "val", "calibration", "test")
VALID_SPLITS = COMMERCIAL_SPLITS


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {manifest_path}")

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("O manifesto não possui cabeçalho CSV.")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "Colunas obrigatórias ausentes: " + ", ".join(sorted(missing))
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]

    if not rows:
        raise ValueError("O manifesto está vazio.")
    return rows


def resolve_manifest_path(
    row: dict[str, str],
    column: str,
    manifest_path: str | Path,
    image_root: str | Path | None = None,
) -> Path:
    path = Path(row[column])
    if path.is_absolute():
        return path
    base = Path(image_root) if image_root else Path(manifest_path).parent
    return (base / path).resolve()


def resolve_image_path(
    row: dict[str, str],
    manifest_path: str | Path,
    image_root: str | Path | None = None,
) -> Path:
    return resolve_manifest_path(row, "image_path", manifest_path, image_root)


def validate_rows(
    rows: Iterable[dict[str, str]],
    *,
    manifest_path: str | Path | None = None,
    image_root: str | Path | None = None,
    check_images: bool = False,
    additional_image_columns: Iterable[str] = (),
    additional_file_columns: Iterable[str] = (),
) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Nenhuma amostra foi fornecida.")

    event_weights: dict[str, float] = {}
    seen_images: set[str] = set()
    errors: list[str] = []

    for line_number, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if not row.get(column, "").strip():
                errors.append(f"linha {line_number}: {column} vazio")

        try:
            weight = float(row.get("weight_kg", ""))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"linha {line_number}: weight_kg deve ser positivo")
            continue

        event_id = row.get("event_id", "")
        previous_weight = event_weights.setdefault(event_id, weight)
        if not math.isclose(previous_weight, weight, abs_tol=1e-6):
            errors.append(
                f"linha {line_number}: evento {event_id!r} possui pesos conflitantes"
            )

        image_key = row.get("image_path", "")
        if image_key in seen_images:
            errors.append(f"linha {line_number}: image_path duplicado: {image_key}")
        seen_images.add(image_key)

        split = row.get("split", "")
        if split and split not in VALID_SPLITS:
            errors.append(f"linha {line_number}: split inválido: {split!r}")

        if check_images:
            if manifest_path is None:
                raise ValueError("manifest_path é obrigatório para verificar imagens")
            for image_column in ("image_path", *additional_image_columns):
                if not row.get(image_column, "").strip():
                    errors.append(f"linha {line_number}: {image_column} vazio")
                    continue
                image_path = resolve_manifest_path(
                    row, image_column, manifest_path, image_root
                )
                if not image_path.is_file():
                    errors.append(
                        f"linha {line_number}: imagem não encontrada: {image_path}"
                    )
                else:
                    try:
                        from PIL import Image

                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            f"linha {line_number}: imagem inválida {image_path}: {exc}"
                        )
            for file_column in additional_file_columns:
                if not row.get(file_column, "").strip():
                    errors.append(f"linha {line_number}: {file_column} vazio")
                    continue
                file_path = resolve_manifest_path(
                    row, file_column, manifest_path, image_root
                )
                if not file_path.is_file():
                    errors.append(
                        f"linha {line_number}: arquivo não encontrado: {file_path}"
                    )

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        suffix = "\n- ..." if len(errors) > 20 else ""
        raise ValueError(f"Manifesto inválido:\n{preview}{suffix}")

    if all(row.get("split", "") for row in rows):
        assert_no_animal_leakage(rows)


def assert_no_animal_leakage(rows: Iterable[dict[str, str]]) -> None:
    animal_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split", "")
        if split:
            animal_splits[row["animal_id"]].add(split)

    leaked = {
        animal_id: sorted(splits)
        for animal_id, splits in animal_splits.items()
        if len(splits) > 1
    }
    if leaked:
        examples = list(leaked.items())[:10]
        details = ", ".join(f"{animal}: {splits}" for animal, splits in examples)
        raise ValueError(f"Vazamento entre splits detectado por animal: {details}")


def _split_counts(size: int, ratios: tuple[float, ...]) -> list[int]:
    exact = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = size - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (exact[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _grouped_split(
    rows: Iterable[dict[str, str]],
    *,
    split_names: tuple[str, ...],
    ratios: tuple[float, ...],
    seed: int = 42,
    stratify_bins: int = 5,
) -> list[dict[str, str]]:
    rows = [dict(row) for row in rows]
    if len(split_names) != len(ratios) or len(set(split_names)) != len(split_names):
        raise ValueError("Nomes e proporções de split são incompatíveis.")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("As proporções devem ser positivas e somar 1.0.")

    weights_by_animal: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        weights_by_animal[row["animal_id"]].append(float(row["weight_kg"]))

    animals = sorted(
        ((animal, median(weights)) for animal, weights in weights_by_animal.items()),
        key=lambda item: item[1],
    )
    if len(animals) < len(split_names):
        raise ValueError(
            f"São necessários ao menos {len(split_names)} animais para criar "
            "os splits."
        )

    # Cada estrato precisa ser grande o bastante para representar também a
    # menor partição. Em datasets pequenos, reduzimos automaticamente os bins.
    number_of_bins = max(
        1,
        min(stratify_bins, math.floor(len(animals) * min(ratios))),
    )
    bins: list[list[tuple[str, float]]] = [[] for _ in range(number_of_bins)]
    for rank, item in enumerate(animals):
        bin_index = min(number_of_bins - 1, rank * number_of_bins // len(animals))
        bins[bin_index].append(item)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for items in bins:
        rng.shuffle(items)
        counts = _split_counts(len(items), ratios)
        cursor = 0
        for split, count in zip(split_names, counts, strict=True):
            for animal_id, _ in items[cursor : cursor + count]:
                assignments[animal_id] = split
            cursor += count

    # Em conjuntos muito pequenos, o arredondamento por estrato pode esvaziar
    # uma partição. Move um grupo da maior partição para manter o contrato.
    for missing_split in split_names:
        if missing_split in assignments.values():
            continue
        counts_by_split = {
            split: list(assignments.values()).count(split) for split in split_names
        }
        donor = max(counts_by_split, key=counts_by_split.get)
        donor_animals = sorted(
            animal for animal, split in assignments.items() if split == donor
        )
        if len(donor_animals) <= 1:
            raise ValueError("Não foi possível criar três splits não vazios.")
        assignments[donor_animals[-1]] = missing_split

    for row in rows:
        row["split"] = assignments[row["animal_id"]]
    assert_no_animal_leakage(rows)
    return rows


def grouped_split(
    rows: Iterable[dict[str, str]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_bins: int = 5,
) -> list[dict[str, str]]:
    """Cria o split histórico de pesquisa, sempre agrupado por animal."""
    return _grouped_split(
        rows,
        split_names=BASELINE_SPLITS,
        ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
        stratify_bins=stratify_bins,
    )


def grouped_commercial_split(
    rows: Iterable[dict[str, str]],
    *,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    calibration_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_bins: int = 5,
) -> list[dict[str, str]]:
    """Reserva calibração independente sem vazar animais entre conjuntos."""
    return _grouped_split(
        rows,
        split_names=COMMERCIAL_SPLITS,
        ratios=(train_ratio, val_ratio, calibration_ratio, test_ratio),
        seed=seed,
        stratify_bins=stratify_bins,
    )


def write_manifest(rows: Iterable[dict[str, str]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Não é possível gravar um manifesto vazio.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
