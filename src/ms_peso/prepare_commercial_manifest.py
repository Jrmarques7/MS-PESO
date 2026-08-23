from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from ms_peso.artifacts import save_json
from ms_peso.collection_snapshot import (
    verify_seal_report,
    verify_snapshot_integrity,
)
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import (
    COMMERCIAL_SPLITS,
    grouped_commercial_split,
    read_manifest,
    validate_rows,
    write_manifest,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria splits comerciais a partir de um snapshot selado."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--snapshot-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", default="data/raw/pilot", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stratify-bins", type=int, default=5)
    parser.add_argument(
        "--output-report",
        default="artifacts/commercial_split/report.json",
        type=Path,
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    payload: dict[str, object]
    report_already_exists = args.output_report.exists()
    try:
        if report_already_exists:
            raise FileExistsError(
                "O relatório de split já existe; escolha outro --output-report."
            )
        if args.output.exists():
            raise FileExistsError(
                "O manifesto dividido já existe; escolha outro --output."
            )
        rows = read_manifest(args.input)
        if any(row.get("split") for row in rows):
            raise ValueError("O snapshot de origem já contém split.")
        validate_rows(
            rows,
            manifest_path=args.input,
            image_root=args.image_root,
            check_images=True,
        )
        verify_snapshot_integrity(
            rows,
            manifest_path=args.input,
            image_root=args.image_root,
        )
        snapshot_id = verify_seal_report(
            args.snapshot_report,
            sealed_manifest_path=args.input,
            rows=rows,
        )
        split_rows = grouped_commercial_split(
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            calibration_ratio=args.calibration_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stratify_bins=args.stratify_bins,
        )
        write_manifest(split_rows, args.output)
        image_counts = Counter(row["split"] for row in split_rows)
        animal_counts = Counter(
            (row["animal_id"], row["split"]) for row in split_rows
        )
        animals_per_split = Counter(split for _, split in animal_counts)
        weights_per_split = {
            split: [
                float(row["weight_kg"])
                for row in split_rows
                if row["split"] == split
            ]
            for split in COMMERCIAL_SPLITS
        }
        payload = {
            "status": "passed",
            "snapshot_id": snapshot_id,
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "calibration": args.calibration_ratio,
                "test": args.test_ratio,
            },
            "splits": {
                split: {
                    "animals": animals_per_split[split],
                    "images": image_counts[split],
                    "weight_kg": {
                        "min": min(weights_per_split[split]),
                        "max": max(weights_per_split[split]),
                        "mean": round(fmean(weights_per_split[split]), 4),
                    },
                }
                for split in COMMERCIAL_SPLITS
            },
            "provenance": {
                "source_manifest_sha256": calculate_sha256(args.input),
                "source_snapshot_report_sha256": calculate_sha256(
                    args.snapshot_report
                ),
                "output_manifest_path": str(args.output.resolve()),
                "output_manifest_sha256": calculate_sha256(args.output),
            },
            "errors": [],
            "warnings": [],
        }
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        payload = {
            "status": "rejected",
            "snapshot_id": None,
            "errors": [str(exc)],
            "warnings": [],
        }

    if not report_already_exists:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output_report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
