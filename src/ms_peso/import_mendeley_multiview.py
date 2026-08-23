from __future__ import annotations

import argparse
from collections import Counter

from ms_peso.importers.mendeley_multiview import (
    SUPPORTED_ANGLES,
    build_multiview_manifest_rows,
)
from ms_peso.manifest import validate_rows, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um manifesto de revisão do dataset multivista Mendeley."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument(
        "--angles",
        nargs="+",
        choices=SUPPORTED_ANGLES,
        default=["angle_1"],
    )
    parser.add_argument(
        "--acknowledge-unverified-source-labels",
        action="store_true",
        help="Reconhece que os pesos ainda exigem auditoria independente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_multiview_manifest_rows(
        args.dataset_root,
        image_root=args.image_root,
        angles=args.angles,
        acknowledge_unverified_source_labels=(
            args.acknowledge_unverified_source_labels
        ),
    )
    validate_rows(
        rows,
        manifest_path=args.output,
        image_root=args.image_root,
        check_images=True,
    )
    write_manifest(rows, args.output)
    weights = [float(row["weight_kg"]) for row in rows]
    views = Counter(row["view"] for row in rows)
    print(f"Manifesto de revisão gravado em {args.output}")
    print(
        f"Animais: {len({row['animal_id'] for row in rows})}; "
        f"imagens: {len(rows)}; vistas: {dict(views)}"
    )
    print(f"Faixa de peso declarada: {min(weights):.1f}–{max(weights):.1f} kg")
    print("Estado: QUARENTENA; não usar como conjunto comercial aprovado.")


if __name__ == "__main__":
    main()
