from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from ms_peso.artifacts import save_json, save_predictions
from ms_peso.collection_snapshot import verify_snapshot_integrity
from ms_peso.commercial_calibration import (
    validate_commercial_calibration_contract,
)
from ms_peso.commercial_data import validate_commercial_data_contract
from ms_peso.config import load_yaml_config
from ms_peso.conformal import (
    calibrate_grouped_absolute_residuals,
    conformal_quantile_rank,
)
from ms_peso.dataset import CattleWeightDataset
from ms_peso.evaluation import evaluate_model
from ms_peso.inference import resolve_device
from ms_peso.manifest import read_manifest, validate_rows, write_manifest
from ms_peso.model import build_model
from ms_peso.reproducibility import set_global_seed


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra o intervalo conformal do candidato comercial."
    )
    parser.add_argument(
        "--config", default="configs/efficientnet_b0_commercial_calibration.yaml"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    data_config = config["data"]
    manifest_path = Path(data_config["manifest"])
    split_report_path = Path(data_config["split_report"])
    image_root = data_config.get("image_root")
    output_dir = Path(config["output"]["directory"])
    rows = read_manifest(manifest_path)
    data_contract = validate_commercial_data_contract(
        rows,
        manifest_path=manifest_path,
        split_report_path=split_report_path,
    )
    checkpoint = validate_commercial_calibration_contract(
        config,
        data_contract,
        output_dir=output_dir,
    )

    selected_view = data_config["view"]
    calibration_rows = [
        row
        for row in rows
        if row["split"] == "calibration" and row.get("view") == selected_view
    ]
    if not calibration_rows:
        raise ValueError("Nenhuma imagem de calibração para a vista configurada.")
    conformal_quantile_rank(
        len({row["animal_id"] for row in calibration_rows}),
        target_coverage=checkpoint.target_coverage,
    )
    validate_rows(
        calibration_rows,
        manifest_path=manifest_path,
        image_root=image_root,
        check_images=True,
    )
    verify_snapshot_integrity(
        calibration_rows,
        manifest_path=manifest_path,
        image_root=image_root,
    )

    dataset = CattleWeightDataset(
        calibration_rows,
        manifest_path=manifest_path,
        image_root=image_root,
        image_size=checkpoint.image_size,
        training=False,
        target_mean=checkpoint.target_mean,
        target_std=checkpoint.target_std,
    )
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        checkpoint.architecture,
        pretrained=False,
        dropout=checkpoint.dropout,
    ).to(device)
    model.load_state_dict(checkpoint.state_dict, strict=True)
    result = evaluate_model(
        model,
        loader,
        device,
        target_mean=checkpoint.target_mean,
        target_std=checkpoint.target_std,
    )
    group_ids = [calibration_rows[index]["animal_id"] for index in result.indices]
    calibration = calibrate_grouped_absolute_residuals(
        result.targets,
        result.predictions,
        group_ids,
        target_coverage=checkpoint.target_coverage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        calibration_rows, output_dir / "resolved_calibration_manifest.csv"
    )
    save_predictions(
        output_dir / "predictions_calibration.csv",
        calibration_rows,
        targets=result.targets,
        predictions=result.predictions,
        indices=result.indices,
    )
    report = {
        "status": "calibrated",
        "workflow": "commercial_calibration",
        "method": "split_conformal_grouped_absolute_residual",
        "grouping": "animal_id_max_absolute_residual",
        "architecture": checkpoint.architecture,
        "image_size": checkpoint.image_size,
        "input_view": checkpoint.input_view,
        "device": str(device),
        "seed": seed,
        "target_coverage": calibration.target_coverage,
        "alpha": calibration.alpha,
        "interval_radius_kg": calibration.radius_kg,
        "quantile_rank": calibration.quantile_rank,
        "number_of_calibration_animals": calibration.number_of_groups,
        "number_of_calibration_images": len(calibration_rows),
        "empirical_group_coverage": calibration.empirical_group_coverage,
        "test_evaluated": False,
        "promotion_status": "not_promoted",
        "commercial_use_allowed": False,
        "source_checkpoint_sha256": checkpoint.sha256,
        "source_snapshot_id": data_contract.snapshot_id,
        "source_manifest_sha256": data_contract.manifest_sha256,
        "source_split_report_sha256": data_contract.split_report_sha256,
    }
    save_json(output_dir / "calibration.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
