from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ms_peso.importers.cowdb import SUPPORTED_VIEWS, build_cowdb_manifest_rows
from ms_peso.manifest import validate_rows, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte uma cópia local do CowDB em manifesto MS-PESO."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=SUPPORTED_VIEWS,
        default=["left"],
    )
    parser.add_argument(
        "--measurements",
        help="Padrão: <dataset-root>/Manual_measurements.xlsx",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Inclui o caminho da profundidade sincronizada no manifesto.",
    )
    parser.add_argument(
        "--include-point-cloud",
        action="store_true",
        help="Inclui o caminho da nuvem PLY sincronizada no manifesto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    measurements_path = (
        Path(args.measurements)
        if args.measurements
        else dataset_root / "Manual_measurements.xlsx"
    )
    rows = build_cowdb_manifest_rows(
        dataset_root,
        measurements_path,
        image_root=args.image_root,
        views=args.views,
        include_depth=args.include_depth,
        include_point_cloud=args.include_point_cloud,
    )
    validate_rows(
        rows,
        manifest_path=args.output,
        image_root=args.image_root,
        check_images=True,
        additional_image_columns=("depth_image_path",) if args.include_depth else (),
        additional_file_columns=("point_cloud_path",)
        if args.include_point_cloud
        else (),
    )
    write_manifest(rows, args.output)

    views = Counter(row["view"] for row in rows)
    animals = {row["animal_id"] for row in rows}
    weights = [float(row["weight_kg"]) for row in rows]
    print(f"Manifesto CowDB gravado em {args.output}")
    print(f"Animais: {len(animals)}; imagens: {len(rows)}; vistas: {dict(views)}")
    print(f"Faixa de peso: {min(weights):.1f}–{max(weights):.1f} kg")


if __name__ == "__main__":
    main()
