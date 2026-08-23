from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ms_peso.dataset import CattleWeightDataset
from ms_peso.depth_crop import build_training_background, load_depth_image
from ms_peso.evaluation import evaluate_model
from ms_peso.foreground_geometry import extract_foreground_geometry
from ms_peso.manifest import (
    read_manifest,
    resolve_manifest_path,
    validate_rows,
)
from ms_peso.metrics import regression_metrics
from ms_peso.model import build_model
from ms_peso.point_cloud import read_organized_ply_xyz


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita altura física de PLY sem consultar rótulos de teste."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-image-column", default="depth_image_path")
    parser.add_argument("--point-cloud-column", default="point_cloud_path")
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--background-percentile", type=float, default=100.0)
    parser.add_argument("--foreground-margin-mm", type=float, default=150.0)
    parser.add_argument("--max-depth-mm", type=float, default=6000.0)
    return parser.parse_args(args)


def pearson_correlation(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Correlação exige vetores do mesmo tamanho com duas amostras.")
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        raise ValueError("Correlação exige variação nos dois vetores.")
    return float(np.corrcoef(first, second)[0, 1])


def fit_height_regression(
    heights: list[float], weights: list[float]
) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(heights)), heights))
    intercept, slope = np.linalg.lstsq(design, weights, rcond=None)[0]
    return float(intercept), float(slope)


def predict_height_regression(
    heights: list[float], *, intercept: float, slope: float
) -> list[float]:
    return (intercept + slope * np.asarray(heights)).tolist()


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": float(np.mean(values)),
        "maximum": max(values),
    }


def _evaluate_reference(
    checkpoint_path: str | Path,
    validation_rows: list[dict[str, str]],
    validation_heights: list[float],
    *,
    manifest_path: Path,
    image_root: str | Path | None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        checkpoint["architecture"],
        pretrained=False,
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    image_size = int(checkpoint["config"]["data"]["image_size"])
    dataset = CattleWeightDataset(
        validation_rows,
        manifest_path=manifest_path,
        image_root=image_root,
        image_size=image_size,
        training=False,
        target_mean=float(checkpoint["target_mean"]),
        target_std=float(checkpoint["target_std"]),
    )
    evaluation = evaluate_model(
        model,
        DataLoader(dataset, batch_size=32),
        torch.device("cpu"),
        target_mean=float(checkpoint["target_mean"]),
        target_std=float(checkpoint["target_std"]),
    )
    ordered_heights = [validation_heights[index] for index in evaluation.indices]
    residuals = [
        target - prediction
        for target, prediction in zip(
            evaluation.targets, evaluation.predictions, strict=True
        )
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": evaluation.metrics,
        "height_residual_correlation": pearson_correlation(
            ordered_heights, residuals
        ),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    validate_rows(
        rows,
        manifest_path=manifest_path,
        image_root=args.image_root,
        check_images=True,
        additional_image_columns=(args.depth_image_column,),
        additional_file_columns=(args.point_cloud_column,),
    )
    audited_rows = [row for row in rows if row.get("split") in {"train", "val"}]
    if {row.get("split") for row in audited_rows} != {"train", "val"}:
        raise ValueError("A auditoria exige partições train e val não vazias.")
    background = build_training_background(
        rows,
        manifest_path=manifest_path,
        image_root=args.image_root,
        depth_image_column=args.depth_image_column,
        percentile=args.background_percentile,
    )

    heights_by_split: dict[str, list[float]] = defaultdict(list)
    weights_by_split: dict[str, list[float]] = defaultdict(list)
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    quality: dict[str, list[float]] = defaultdict(list)
    for row in audited_rows:
        depth = load_depth_image(
            resolve_manifest_path(
                row, args.depth_image_column, manifest_path, args.image_root
            )
        )
        xyz = read_organized_ply_xyz(
            resolve_manifest_path(
                row, args.point_cloud_column, manifest_path, args.image_root
            )
        )
        geometry = extract_foreground_geometry(
            depth,
            background,
            xyz,
            margin_mm=args.foreground_margin_mm,
            max_depth_mm=args.max_depth_mm,
        )
        split = row["split"]
        heights_by_split[split].append(geometry.height_m)
        weights_by_split[split].append(float(row["weight_kg"]))
        rows_by_split[split].append(row)
        quality["point_count"].append(float(geometry.point_count))
        quality["mask_fraction"].append(geometry.mask_fraction)
        quality["box_height_fraction"].append(geometry.box_height_fraction)

    intercept, slope = fit_height_regression(
        heights_by_split["train"], weights_by_split["train"]
    )
    report: dict[str, object] = {
        "guardrail": "test excluído da extração e da análise",
        "samples": {split: len(rows_by_split[split]) for split in ("train", "val")},
        "parameters": {
            "background_percentile": args.background_percentile,
            "foreground_margin_mm": args.foreground_margin_mm,
            "max_depth_mm": args.max_depth_mm,
            "height_quantiles": [0.05, 0.95],
        },
        "height_m": {
            split: {
                **_summarize(heights_by_split[split]),
                "weight_correlation": pearson_correlation(
                    heights_by_split[split], weights_by_split[split]
                ),
            }
            for split in ("train", "val")
        },
        "height_only_regression": {
            "trained_on": "train",
            "intercept": intercept,
            "slope": slope,
            "metrics": {
                split: regression_metrics(
                    weights_by_split[split],
                    predict_height_regression(
                        heights_by_split[split], intercept=intercept, slope=slope
                    ),
                )
                for split in ("train", "val")
            },
        },
        "quality": {name: _summarize(values) for name, values in quality.items()},
    }
    if args.reference_checkpoint:
        report["reference_validation"] = _evaluate_reference(
            args.reference_checkpoint,
            rows_by_split["val"],
            heights_by_split["val"],
            manifest_path=manifest_path,
            image_root=args.image_root,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
