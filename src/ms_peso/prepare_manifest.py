from __future__ import annotations

import argparse
from collections import Counter

from ms_peso.manifest import (
    grouped_split,
    read_manifest,
    validate_rows,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida e divide um manifesto sem vazar animais entre conjuntos."
    )
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", required=True, help="CSV de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stratify-bins", type=int, default=5)
    parser.add_argument("--image-root")
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input)
    validate_rows(
        rows,
        manifest_path=args.input,
        image_root=args.image_root,
        check_images=args.check_images,
    )
    rows = grouped_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_bins=args.stratify_bins,
    )
    write_manifest(rows, args.output)

    image_counts = Counter(row["split"] for row in rows)
    animal_counts = Counter((row["animal_id"], row["split"]) for row in rows)
    animals_per_split = Counter(split for _, split in animal_counts)
    print(f"Manifesto gravado em {args.output}")
    for split in ("train", "val", "test"):
        print(
            f"{split}: {animals_per_split[split]} animais, "
            f"{image_counts[split]} imagens"
        )


if __name__ == "__main__":
    main()
