from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_peso.artifacts import save_json, save_predictions
from ms_peso.baselines import evaluate_mean_baseline
from ms_peso.manifest import read_manifest, validate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o baseline B0 usando a média do conjunto de treino."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="artifacts/baseline_mean")
    parser.add_argument("--image-root")
    parser.add_argument("--view")
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    validate_rows(
        rows,
        manifest_path=args.manifest,
        image_root=args.image_root,
        check_images=args.check_images,
    )
    if args.view:
        rows = [row for row in rows if row.get("view") == args.view]
    if not rows:
        raise ValueError("Nenhuma amostra corresponde aos filtros informados.")

    missing_splits = [row["animal_id"] for row in rows if not row.get("split")]
    if missing_splits:
        raise ValueError("O baseline B0 requer um manifesto com split preenchido.")
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    result = evaluate_mean_baseline(
        [float(row["weight_kg"]) for row in train_rows],
        [float(row["weight_kg"]) for row in test_rows],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline": "training_mean",
        "manifest": str(Path(args.manifest)),
        "view": args.view,
        "training_animals": len({row["animal_id"] for row in train_rows}),
        "test_animals": len({row["animal_id"] for row in test_rows}),
        "training_mean_kg": result.training_mean_kg,
        "metrics": result.metrics,
    }
    save_json(output_dir / "metrics.json", report)
    save_predictions(
        output_dir / "predictions_test.csv",
        test_rows,
        targets=result.targets,
        predictions=result.predictions,
        indices=list(range(len(test_rows))),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
