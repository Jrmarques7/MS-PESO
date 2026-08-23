from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from ms_peso.integrity import calculate_content_sha256, calculate_sha256
from ms_peso.manifest import resolve_image_path


@dataclass(frozen=True)
class ImageFingerprint:
    image_path: str
    sha256: str
    dhash: str


@dataclass(frozen=True)
class DuplicateMatch:
    first_image_path: str
    second_image_path: str
    hamming_distance: int

    def to_dict(self) -> dict[str, object]:
        return {
            "first_image_path": self.first_image_path,
            "second_image_path": self.second_image_path,
            "hamming_distance": self.hamming_distance,
        }


@dataclass(frozen=True)
class CollectionSnapshot:
    snapshot_id: str
    rows: tuple[dict[str, str], ...]
    exact_duplicates: tuple[DuplicateMatch, ...]
    near_duplicates: tuple[DuplicateMatch, ...]

    @property
    def valid(self) -> bool:
        return not self.exact_duplicates

    def to_report(self) -> dict[str, object]:
        errors = []
        warnings = []
        if self.exact_duplicates:
            errors.append(
                "Duplicatas exatas foram encontradas; revise a seleção de frames."
            )
        if self.near_duplicates:
            warnings.append(
                "Possíveis duplicatas visuais exigem revisão humana antes do treino."
            )
        return {
            "status": "passed" if self.valid else "rejected",
            "snapshot_id": self.snapshot_id,
            "summary": {
                "images": len(self.rows),
                "exact_duplicate_pairs": len(self.exact_duplicates),
                "near_duplicate_pairs": len(self.near_duplicates),
            },
            "exact_duplicates": [
                match.to_dict() for match in self.exact_duplicates
            ],
            "near_duplicates": [match.to_dict() for match in self.near_duplicates],
            "errors": errors,
            "warnings": warnings,
        }


def calculate_image_dhash(path: str | Path) -> str:
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            gray = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(gray, dtype=np.int16)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Arquivo de imagem inválido: {image_path}") from exc
    comparisons = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def dhash_distance(first: str, second: str) -> int:
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("dHash deve ser hexadecimal.") from exc


def _duplicate_pairs(
    fingerprints: list[ImageFingerprint], *, near_distance: int
) -> tuple[tuple[DuplicateMatch, ...], tuple[DuplicateMatch, ...]]:
    exact: list[DuplicateMatch] = []
    near: list[DuplicateMatch] = []
    for index, first in enumerate(fingerprints):
        for second in fingerprints[index + 1 :]:
            distance = dhash_distance(first.dhash, second.dhash)
            match = DuplicateMatch(
                first_image_path=first.image_path,
                second_image_path=second.image_path,
                hamming_distance=distance,
            )
            if first.sha256 == second.sha256:
                exact.append(match)
            elif distance <= near_distance:
                near.append(match)
    return tuple(exact), tuple(near)


def _canonical_rows(rows: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.get("farm_id", ""),
                row.get("animal_id", ""),
                row.get("event_id", ""),
                row.get("view", ""),
                row.get("image_path", ""),
            ),
        )
    )


def build_collection_snapshot(
    rows: list[dict[str, str]],
    *,
    manifest_path: str | Path,
    image_root: str | Path | None,
    near_duplicate_hamming_distance: int,
) -> CollectionSnapshot:
    if not rows:
        raise ValueError("Nenhuma imagem foi fornecida para o snapshot.")
    if not 0 <= near_duplicate_hamming_distance <= 64:
        raise ValueError("Distância de duplicidade deve estar entre 0 e 64.")

    fingerprints: list[ImageFingerprint] = []
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = resolve_image_path(row, manifest_path, image_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        fingerprint = ImageFingerprint(
            image_path=row["image_path"],
            sha256=calculate_sha256(image_path),
            dhash=calculate_image_dhash(image_path),
        )
        fingerprints.append(fingerprint)
        enriched_rows.append(
            {
                **row,
                "image_sha256": fingerprint.sha256,
                "image_dhash": fingerprint.dhash,
            }
        )

    canonical_rows = _canonical_rows(enriched_rows)
    canonical_json = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    exact, near = _duplicate_pairs(
        fingerprints,
        near_distance=near_duplicate_hamming_distance,
    )
    return CollectionSnapshot(
        snapshot_id=calculate_content_sha256(canonical_json),
        rows=canonical_rows,
        exact_duplicates=exact,
        near_duplicates=near,
    )
