from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ms_peso.depth_crop import (
    build_training_background,
    detect_depth_foreground_box,
    load_depth_image,
    render_depth_guided_rgb,
)
from ms_peso.manifest import (
    read_manifest,
    resolve_image_path,
    resolve_manifest_path,
    validate_rows,
    write_manifest,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria recortes RGB guiados pela profundidade do treino."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--depth-image-column", default="depth_image_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--background-percentile", type=float, default=100.0)
    parser.add_argument("--foreground-margin-mm", type=float, default=150.0)
    parser.add_argument("--max-depth-mm", type=float, default=6000.0)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--minimum-box-area", type=float, default=0.50)
    parser.add_argument(
        "--output-mode",
        choices=("crop", "masked_canvas"),
        default="crop",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    try:
        output_relative = output_dir.relative_to(image_root)
    except ValueError as exc:
        raise ValueError("output-dir deve estar dentro de image-root.") from exc

    rows = read_manifest(manifest_path)
    validate_rows(
        rows,
        manifest_path=manifest_path,
        image_root=image_root,
        check_images=True,
        additional_image_columns=(args.depth_image_column,),
    )
    background = build_training_background(
        rows,
        manifest_path=manifest_path,
        image_root=image_root,
        depth_image_column=args.depth_image_column,
        percentile=args.background_percentile,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "background_depth.npy", background)

    cropped_rows: list[dict[str, str]] = []
    box_areas: list[float] = []
    for row in rows:
        depth_path = resolve_manifest_path(
            row, args.depth_image_column, manifest_path, image_root
        )
        box = detect_depth_foreground_box(
            load_depth_image(depth_path),
            background,
            margin_mm=args.foreground_margin_mm,
            max_depth_mm=args.max_depth_mm,
            padding=args.padding,
        )
        rgb_path = resolve_image_path(row, manifest_path, image_root)
        output_name = f"{row['animal_id']}_{row.get('view', 'unknown')}.png"
        output_path = output_dir / output_name
        box_area = (box.right - box.left) * (box.bottom - box.top)
        if box_area < args.minimum_box_area:
            raise ValueError(
                f"Recorte de {row['animal_id']} ocupa apenas {box_area:.1%} "
                f"da imagem; mínimo exigido: {args.minimum_box_area:.1%}."
            )
        with Image.open(rgb_path) as image:
            render_depth_guided_rgb(
                image, box, output_mode=args.output_mode
            ).save(output_path)
        box_areas.append(box_area)
        cropped_rows.append(
            {
                **row,
                "image_path": (output_relative / output_name).as_posix(),
                "source_image_path": row["image_path"],
                "crop_left": f"{box.left:.6f}",
                "crop_top": f"{box.top:.6f}",
                "crop_right": f"{box.right:.6f}",
                "crop_bottom": f"{box.bottom:.6f}",
                "crop_method": (
                    "training_depth_background_subtraction_" + args.output_mode
                ),
            }
        )

    write_manifest(cropped_rows, args.output_manifest)
    report = {
        "samples": len(cropped_rows),
        "background_rows": sum(row.get("split") == "train" for row in rows),
        "box_area_fraction": {
            "minimum": min(box_areas),
            "mean": sum(box_areas) / len(box_areas),
            "maximum": max(box_areas),
        },
        "parameters": {
            "background_percentile": args.background_percentile,
            "foreground_margin_mm": args.foreground_margin_mm,
            "max_depth_mm": args.max_depth_mm,
            "padding": args.padding,
            "minimum_box_area": args.minimum_box_area,
            "output_mode": args.output_mode,
        },
    }
    (output_dir / "crop_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
