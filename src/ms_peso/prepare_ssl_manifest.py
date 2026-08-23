from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import read_manifest, resolve_image_path

SSL_FIELDS = (
    "image_path",
    "ssl_sample_id",
    "source_dataset",
    "source_license",
    "source_animal_id",
    "source_view",
    "source_split",
    "source_manifest",
    "original_image_path",
    "image_sha256",
)


def build_ssl_manifest_rows(
    manifest_paths: Iterable[str | Path],
    *,
    image_root: str | Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Combina apenas linhas sem split ou de treino e remove conteúdo repetido."""
    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("Informe ao menos um manifesto de origem.")

    candidates: list[dict[str, str]] = []
    skipped_by_split = 0
    for manifest_path in paths:
        for row in read_manifest(manifest_path):
            split = row.get("split", "")
            if split and split != "train":
                skipped_by_split += 1
                continue
            image_path = resolve_image_path(row, manifest_path, image_root)
            if not image_path.is_file():
                raise FileNotFoundError(f"Imagem SSL não encontrada: {image_path}")
            candidates.append(
                {
                    "image_path": row["image_path"],
                    "source_dataset": row.get("source_dataset", "unknown"),
                    "source_license": row.get("source_license", ""),
                    "source_animal_id": row.get(
                        "source_animal_id", row.get("animal_id", "")
                    ),
                    "source_view": row.get("view", ""),
                    "source_split": split or "unsplit",
                    "source_manifest": manifest_path.as_posix(),
                    "original_image_path": row.get(
                        "source_image_path", row["image_path"]
                    ),
                    "image_sha256": calculate_sha256(image_path),
                }
            )

    rows: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    duplicates = 0
    for candidate in candidates:
        digest = candidate["image_sha256"]
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        rows.append(
            {
                "image_path": candidate["image_path"],
                "ssl_sample_id": f"ssl_{len(rows) + 1:05d}",
                **{key: candidate[key] for key in SSL_FIELDS[2:]},
            }
        )
    if len(rows) < 2:
        raise ValueError("São necessárias ao menos duas imagens SSL únicas.")
    stats = {
        "candidates": len(candidates),
        "retained": len(rows),
        "duplicates_removed": duplicates,
        "non_train_rows_skipped": skipped_by_split,
    }
    return rows, stats


def write_ssl_manifest(rows: Iterable[dict[str, str]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SSL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_ssl_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(SSL_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Manifesto SSL sem colunas: " + ", ".join(sorted(missing))
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("Manifesto SSL vazio.")
    if any(row["source_split"] not in {"train", "unsplit"} for row in rows):
        raise ValueError("Manifesto SSL contém validação ou teste.")
    hashes = [row["image_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Manifesto SSL contém imagens duplicadas.")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combina imagens licenciadas para pré-treinamento sem rótulos."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build_ssl_manifest_rows(args.input, image_root=args.image_root)
    licenses = Counter(row["source_license"] for row in rows)
    if set(licenses) != {"CC_BY_4_0"}:
        raise ValueError(
            f"Fonte SSL sem licença CC BY 4.0 confirmada: {dict(licenses)}"
        )
    write_ssl_manifest(rows, args.output)
    print(f"Manifesto SSL gravado em {args.output}")
    print(f"Auditoria: {stats}; licenças: {dict(licenses)}")


if __name__ == "__main__":
    main()
