from __future__ import annotations

import argparse

from ms_peso.manifest import read_manifest, validate_rows, write_manifest
from ms_peso.multi_view import pair_views


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combina duas vistas sincronizadas em uma amostra por evento."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--primary-view", default="left")
    parser.add_argument("--secondary-view", default="top")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input)
    validate_rows(
        rows,
        manifest_path=args.input,
        image_root=args.image_root,
        check_images=True,
    )
    paired_rows = pair_views(
        rows,
        primary_view=args.primary_view,
        secondary_view=args.secondary_view,
    )
    validate_rows(
        paired_rows,
        manifest_path=args.input,
        image_root=args.image_root,
        check_images=True,
        additional_image_columns=("secondary_image_path",),
    )
    write_manifest(paired_rows, args.output)
    print(
        f"Manifesto multivista gravado em {args.output}: "
        f"{len(paired_rows)} eventos com {args.primary_view} + {args.secondary_view}."
    )


if __name__ == "__main__":
    main()
