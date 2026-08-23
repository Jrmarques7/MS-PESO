from __future__ import annotations

import argparse
from collections import Counter

from ms_peso.importers.horqin import SUPPORTED_VIEWS, build_horqin_manifest_rows
from ms_peso.manifest import validate_rows, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte o dataset Horqin lateral/traseiro em manifesto MS-PESO."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument(
        "--views", nargs="+", choices=SUPPORTED_VIEWS, default=["side"]
    )
    parser.add_argument(
        "--exclude-known-anomalies",
        action="store_true",
        help="Exclui animais com vista ausente ou resolução anômala.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_horqin_manifest_rows(
        args.dataset_root,
        image_root=args.image_root,
        views=args.views,
        exclude_known_anomalies=args.exclude_known_anomalies,
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
    print(f"Manifesto Horqin gravado em {args.output}")
    print(
        f"Animais: {len({row['animal_id'] for row in rows})}; "
        f"imagens: {len(rows)}; vistas: {dict(views)}"
    )
    print(f"Faixa de peso: {min(weights):.1f}–{max(weights):.1f} kg")


if __name__ == "__main__":
    main()
