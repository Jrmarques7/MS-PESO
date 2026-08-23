from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps

from ms_peso.manifest import (
    read_manifest,
    resolve_image_path,
    validate_rows,
    write_manifest,
)


def prepare_rgb_cache(
    rows: Iterable[dict[str, str]],
    *,
    manifest_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    image_size: int,
) -> list[dict[str, str]]:
    """Materializa o redimensionamento determinístico sem alterar a fonte bruta."""
    if image_size <= 0:
        raise ValueError("image_size deve ser positivo.")
    source_rows = [dict(row) for row in rows]
    validate_rows(source_rows)
    output_directory = Path(output_dir)
    if output_directory.exists() and not output_directory.is_dir():
        raise FileExistsError(f"Destino existe e não é uma pasta: {output_directory}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Destino já contém arquivos e não será sobrescrito: {output_directory}"
        )

    image_base = Path(image_root).resolve()
    try:
        output_directory.resolve().relative_to(image_base)
    except ValueError as exc:
        raise ValueError(
            f"output_dir {output_directory} deve estar dentro de "
            f"image_root {image_base}."
        ) from exc
    output_directory.mkdir(parents=True, exist_ok=True)

    derived_rows: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for index, row in enumerate(source_rows, start=1):
        source_path = resolve_image_path(row, manifest_path, image_root)
        output_name = _output_name(row, index)
        if output_name in seen_names:
            raise ValueError(f"Nome derivado duplicado: {output_name}")
        seen_names.add(output_name)
        output_path = output_directory / output_name
        with Image.open(source_path) as image:
            converted = ImageOps.exif_transpose(image).convert("RGB")
            resized = converted.resize(
                (image_size, image_size), resample=Image.Resampling.BILINEAR
            )
            resized.save(output_path, format="PNG", compress_level=1)

        relative_output = output_path.resolve().relative_to(image_base)
        derived_rows.append(
            {
                **row,
                "image_path": relative_output.as_posix(),
                "source_image_path": row["image_path"],
                "derived_transform": f"resize_bilinear_{image_size}x{image_size}",
                "derived_image_width": str(image_size),
                "derived_image_height": str(image_size),
            }
        )
    return derived_rows


def _output_name(row: dict[str, str], index: int) -> str:
    components = [
        row.get("animal_id", "animal"),
        row.get("event_id", "event"),
        row.get("view", "view"),
        f"{index:04d}",
    ]
    stem = "__".join(re.sub(r"[^a-zA-Z0-9_-]+", "_", item) for item in components)
    return f"{stem}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria cache RGB redimensionado e manifesto derivado."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input)
    derived_rows = prepare_rgb_cache(
        rows,
        manifest_path=args.input,
        image_root=args.image_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
    validate_rows(
        derived_rows,
        manifest_path=args.output,
        image_root=args.image_root,
        check_images=True,
    )
    write_manifest(derived_rows, args.output)
    print(
        f"Cache RGB gravado em {args.output_dir}; "
        f"manifesto: {args.output}; imagens: {len(derived_rows)}"
    )


if __name__ == "__main__":
    main()
