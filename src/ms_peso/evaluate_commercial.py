from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import DataLoader

from ms_peso.artifacts import (
    save_interval_predictions,
    save_json,
    save_json_exclusive,
)
from ms_peso.collection_snapshot import verify_snapshot_integrity
from ms_peso.commercial_data import validate_commercial_data_contract
from ms_peso.commercial_evaluation import validate_commercial_evaluation_contract
from ms_peso.config import load_yaml_config
from ms_peso.dataset import CattleWeightDataset
from ms_peso.evaluation import evaluate_model
from ms_peso.inference import resolve_device
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import read_manifest, validate_rows, write_manifest
from ms_peso.metrics import regression_metrics
from ms_peso.model import build_model
from ms_peso.reproducibility import set_global_seed
from ms_peso.test_evaluation import (
    cluster_bootstrap_metrics,
    evaluate_symmetric_interval_coverage,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia uma única vez o teste do candidato comercial."
    )
    parser.add_argument(
        "--config", default="configs/efficientnet_b0_commercial_evaluation.yaml"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(args)


def _criterion(
    *, observed: float | int, limit: float | int, operator: str, passed: bool
) -> dict[str, object]:
    return {
        "passed": passed,
        "observed": observed,
        "operator": operator,
        "limit": limit,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
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
    contract = validate_commercial_evaluation_contract(
        config,
        data_contract,
        output_dir=output_dir,
    )

    selected_view = data_config["view"]
    test_rows = [
        row
        for row in rows
        if row["split"] == "test" and row.get("view") == selected_view
    ]
    number_of_test_animals = len({row["animal_id"] for row in test_rows})
    if number_of_test_animals < contract.acceptance.minimum_test_animals:
        raise ValueError(
            "Animais de teste da vista selecionada são insuficientes; teste preservado."
        )

    receipt_payload = {
        "status": "test_opened",
        "test_consumed": True,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config_sha256": calculate_sha256(config_path),
        "source_checkpoint_sha256": contract.checkpoint.sha256,
        "source_calibration_sha256": contract.calibration.sha256,
        "interval_method": "split_conformal_grouped_absolute_residual",
        "interval_target_coverage": contract.calibration.target_coverage,
        "interval_radius_kg": contract.calibration.interval_radius_kg,
        "source_snapshot_id": data_contract.snapshot_id,
        "source_manifest_sha256": data_contract.manifest_sha256,
        "source_split_report_sha256": data_contract.split_report_sha256,
        "number_of_test_animals": number_of_test_animals,
        "number_of_test_images": len(test_rows),
        "acceptance_criteria": contract.acceptance.to_dict(),
        "bootstrap_iterations": contract.bootstrap_iterations,
        "warning": (
            "Alterar modelo, calibração ou critérios após este recibo exige um novo "
            "conjunto de teste independente."
        ),
    }
    contract.access_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_exclusive(contract.access_receipt_path, receipt_payload)

    validate_rows(
        test_rows,
        manifest_path=manifest_path,
        image_root=image_root,
        check_images=True,
    )
    verify_snapshot_integrity(
        test_rows,
        manifest_path=manifest_path,
        image_root=image_root,
    )
    checkpoint = contract.checkpoint
    dataset = CattleWeightDataset(
        test_rows,
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
    group_ids = [test_rows[index]["animal_id"] for index in result.indices]
    bootstrap_metrics = cluster_bootstrap_metrics(
        result.targets,
        result.predictions,
        group_ids,
        iterations=contract.bootstrap_iterations,
        seed=seed,
    )
    coverage = evaluate_symmetric_interval_coverage(
        result.targets,
        result.predictions,
        group_ids,
        radius_kg=contract.calibration.interval_radius_kg,
    )
    acceptance = contract.acceptance
    absolute_bias_bound = max(
        abs(bootstrap_metrics["bias_kg"]["lower_95"]),
        abs(bootstrap_metrics["bias_kg"]["upper_95"]),
    )
    checks = {
        "minimum_test_animals": _criterion(
            observed=coverage.number_of_groups,
            limit=acceptance.minimum_test_animals,
            operator=">=",
            passed=coverage.number_of_groups >= acceptance.minimum_test_animals,
        ),
        "maximum_mae_upper_95_kg": _criterion(
            observed=bootstrap_metrics["mae_kg"]["upper_95"],
            limit=acceptance.maximum_mae_upper_95_kg,
            operator="<=",
            passed=(
                bootstrap_metrics["mae_kg"]["upper_95"]
                <= acceptance.maximum_mae_upper_95_kg
            ),
        ),
        "maximum_rmse_upper_95_kg": _criterion(
            observed=bootstrap_metrics["rmse_kg"]["upper_95"],
            limit=acceptance.maximum_rmse_upper_95_kg,
            operator="<=",
            passed=(
                bootstrap_metrics["rmse_kg"]["upper_95"]
                <= acceptance.maximum_rmse_upper_95_kg
            ),
        ),
        "maximum_mape_upper_95_pct": _criterion(
            observed=bootstrap_metrics["mape_pct"]["upper_95"],
            limit=acceptance.maximum_mape_upper_95_pct,
            operator="<=",
            passed=(
                bootstrap_metrics["mape_pct"]["upper_95"]
                <= acceptance.maximum_mape_upper_95_pct
            ),
        ),
        "maximum_absolute_bias_bound_95_kg": _criterion(
            observed=absolute_bias_bound,
            limit=acceptance.maximum_absolute_bias_bound_95_kg,
            operator="<=",
            passed=(
                absolute_bias_bound <= acceptance.maximum_absolute_bias_bound_95_kg
            ),
        ),
        "minimum_group_coverage_lower_95": _criterion(
            observed=coverage.group_coverage_lower_95,
            limit=acceptance.minimum_group_coverage_lower_95,
            operator=">=",
            passed=(
                coverage.group_coverage_lower_95
                >= acceptance.minimum_group_coverage_lower_95
            ),
        ),
        "maximum_interval_radius_kg": _criterion(
            observed=coverage.radius_kg,
            limit=acceptance.maximum_interval_radius_kg,
            operator="<=",
            passed=coverage.radius_kg <= acceptance.maximum_interval_radius_kg,
        ),
    }
    technical_criteria_passed = all(check["passed"] for check in checks.values())
    recommendation = (
        "technical_review_recommended"
        if technical_criteria_passed
        else "technical_rejection"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(test_rows, output_dir / "resolved_test_manifest.csv")
    save_interval_predictions(
        output_dir / "predictions_test.csv",
        test_rows,
        targets=result.targets,
        predictions=result.predictions,
        indices=result.indices,
        radius_kg=coverage.radius_kg,
    )
    image_level_metrics = regression_metrics(result.targets, result.predictions)
    strict_image_level_metrics = {
        key: value if math.isfinite(value) else None
        for key, value in image_level_metrics.items()
    }
    report = {
        "status": "evaluated",
        "workflow": "commercial_evaluation",
        "test_evaluated": True,
        "test_consumed": True,
        "device": str(device),
        "seed": seed,
        "architecture": checkpoint.architecture,
        "image_size": checkpoint.image_size,
        "input_view": checkpoint.input_view,
        "number_of_test_animals": coverage.number_of_groups,
        "number_of_test_images": coverage.number_of_images,
        "image_level_metrics": strict_image_level_metrics,
        "animal_balanced_metrics_bootstrap_95": bootstrap_metrics,
        "interval_method": "split_conformal_grouped_absolute_residual",
        "interval_target_coverage": contract.calibration.target_coverage,
        "interval_coverage": asdict(coverage),
        "acceptance_criteria": acceptance.to_dict(),
        "acceptance_checks": checks,
        "technical_criteria_passed": technical_criteria_passed,
        "technical_recommendation": recommendation,
        "promotion_status": (
            "review_required" if technical_criteria_passed else "technical_rejected"
        ),
        "commercial_use_allowed": False,
        "mandatory_remaining_reviews": [
            "legal_rights_review",
            "external_domain_validation",
            "operational_safety_review",
            "human_promotion_approval",
        ],
        "source_checkpoint_sha256": checkpoint.sha256,
        "source_calibration_sha256": contract.calibration.sha256,
        "source_snapshot_id": data_contract.snapshot_id,
        "source_manifest_sha256": data_contract.manifest_sha256,
        "source_split_report_sha256": data_contract.split_report_sha256,
        "test_access_receipt_sha256": calculate_sha256(
            contract.access_receipt_path
        ),
        "retraining_or_threshold_changes_require_new_test": True,
    }
    save_json(output_dir / "final_test_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
