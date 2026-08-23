from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_peso.collection import load_collection_policy
from ms_peso.collection_video import select_collection_frames
from ms_peso.image_quality import load_image_quality_policy
from ms_peso.service.video_policy import load_video_inference_policy


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona um quadro técnico por evento/vista sem estimar nem "
            "alterar o peso."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--video-root", default="data/raw/pasture", type=Path)
    parser.add_argument("--image-root", default="data", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--collection-policy",
        default="configs/pilot_collection.yaml",
        type=Path,
    )
    parser.add_argument(
        "--quality-policy", default="configs/image_quality.yaml", type=Path
    )
    parser.add_argument(
        "--video-policy", default="configs/video_inference.yaml", type=Path
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    try:
        report = select_collection_frames(
            manifest_path=args.manifest,
            video_root=args.video_root,
            image_root=args.image_root,
            output_directory=args.output_directory,
            collection_policy=load_collection_policy(args.collection_policy),
            quality_policy=load_image_quality_policy(args.quality_policy),
            video_policy=load_video_inference_policy(args.video_policy),
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        report = {"status": "rejected", "error": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
