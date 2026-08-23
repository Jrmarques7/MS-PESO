from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from ms_peso.artifacts import save_checkpoint, save_json, save_predictions
from ms_peso.collection_snapshot import verify_snapshot_integrity
from ms_peso.commercial_training import validate_commercial_fit_contract
from ms_peso.config import load_yaml_config
from ms_peso.dataset import (
    CattleWeightDataset,
    MultiViewCattleWeightDataset,
    RGBDepthCattleWeightDataset,
)
from ms_peso.evaluation import evaluate_model
from ms_peso.manifest import (
    grouped_split,
    read_manifest,
    validate_rows,
    write_manifest,
)
from ms_peso.metrics import regression_metrics
from ms_peso.model import build_model
from ms_peso.reproducibility import set_global_seed
from ms_peso.sampling import inverse_frequency_weights
from ms_peso.training import fit_model


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o baseline RGB do MS-PESO.")
    parser.add_argument("--config", default="configs/baseline_rgb.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    workflow = config["project"].get("workflow", "research")
    if workflow not in {"research", "commercial_fit"}:
        raise ValueError(f"Workflow de treinamento desconhecido: {workflow!r}")

    data_config = config["data"]
    depth_image_column = data_config.get("depth_image_column")
    secondary_image_column = data_config.get("secondary_image_column")
    if depth_image_column and secondary_image_column:
        raise ValueError(
            "Use somente depth_image_column ou secondary_image_column por execução."
        )
    manifest_path = Path(data_config["manifest"])
    image_root = data_config.get("image_root")
    rows = read_manifest(manifest_path)
    output_dir = Path(config["output"]["directory"])
    commercial_contract = None
    if workflow == "commercial_fit":
        split_report = data_config.get("split_report")
        if not split_report:
            raise ValueError("O ajuste comercial exige data.split_report.")
        commercial_contract = validate_commercial_fit_contract(
            config,
            rows,
            manifest_path=manifest_path,
            split_report_path=split_report,
            output_dir=output_dir,
        )
    else:
        validate_rows(
            rows,
            manifest_path=manifest_path,
            image_root=image_root,
            check_images=True,
            additional_image_columns=tuple(
                column
                for column in (depth_image_column, secondary_image_column)
                if column
            ),
        )

    selected_view = data_config.get("view")
    if selected_view and any("view" in row for row in rows):
        rows = [row for row in rows if row.get("view") == selected_view]
        if not rows:
            raise ValueError(
                f"Nenhuma imagem encontrada para a vista {selected_view!r}."
            )

    if workflow == "research" and not all(row.get("split") for row in rows):
        if any(row.get("split") for row in rows):
            raise ValueError("A coluna split está parcialmente preenchida.")
        rows = grouped_split(
            rows,
            train_ratio=float(data_config["train_ratio"]),
            val_ratio=float(data_config["val_ratio"]),
            test_ratio=float(data_config["test_ratio"]),
            seed=seed,
            stratify_bins=int(data_config["stratify_bins"]),
        )

    fit_splits = ("train", "val") if workflow == "commercial_fit" else (
        "train",
        "val",
        "test",
    )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in fit_splits
    }
    if any(not values for values in split_rows.values()):
        required = ", ".join(fit_splits)
        raise ValueError(f"{required} precisam conter ao menos uma amostra.")

    rows_to_open = (
        [row for split in fit_splits for row in split_rows[split]]
        if workflow == "commercial_fit"
        else rows
    )
    if workflow == "commercial_fit":
        validate_rows(
            rows_to_open,
            manifest_path=manifest_path,
            image_root=image_root,
            check_images=True,
            additional_image_columns=tuple(
                column
                for column in (depth_image_column, secondary_image_column)
                if column
            ),
        )
        verify_snapshot_integrity(
            rows_to_open,
            manifest_path=manifest_path,
            image_root=image_root,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest_name = (
        "resolved_fit_manifest.csv"
        if workflow == "commercial_fit"
        else "resolved_manifest.csv"
    )
    write_manifest(rows_to_open, output_dir / resolved_manifest_name)

    train_weights = [float(row["weight_kg"]) for row in split_rows["train"]]
    target_mean = fmean(train_weights)
    target_std = pstdev(train_weights)
    if not math.isfinite(target_std) or target_std < 1e-6:
        raise ValueError("O peso no treino precisa ter variação maior que zero.")

    common_dataset_arguments = {
        "manifest_path": manifest_path,
        "image_root": image_root,
        "image_size": int(data_config["image_size"]),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    if depth_image_column:
        datasets = {
            split: RGBDepthCattleWeightDataset(
                split_rows[split],
                depth_image_column=depth_image_column,
                depth_max_mm=float(data_config["depth_max_mm"]),
                training=split == "train",
                **common_dataset_arguments,
            )
            for split in fit_splits
        }
    elif secondary_image_column:
        datasets = {
            split: MultiViewCattleWeightDataset(
                split_rows[split],
                secondary_image_column=secondary_image_column,
                training=split == "train",
                **common_dataset_arguments,
            )
            for split in fit_splits
        }
    else:
        datasets = {
            split: CattleWeightDataset(
                split_rows[split],
                training=split == "train",
                **common_dataset_arguments,
            )
            for split in fit_splits
        }
    sampling_config = data_config.get("sampling")
    train_sampler = None
    if sampling_config:
        if sampling_config["strategy"] != "inverse_weight_band":
            raise ValueError("Estratégia de amostragem desconhecida")
        sample_weights = inverse_frequency_weights(
            train_weights,
            boundaries=[float(value) for value in sampling_config["boundaries_kg"]],
            power=float(sampling_config["power"]),
        )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loaders = {}
    for split, dataset in datasets.items():
        sampler = train_sampler if split == "train" else None
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            num_workers=int(data_config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = build_model(
        model_config["architecture"],
        pretrained=bool(model_config["pretrained"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    training_config = config["training"]
    criterion = nn.HuberLoss(delta=float(training_config["huber_delta"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    print(
        f"Dispositivo: {device}; amostras: "
        + ", ".join(f"{split}={len(values)}" for split, values in split_rows.items())
    )
    fit_result = fit_model(
        model,
        train_loader=loaders["train"],
        validation_loader=loaders["val"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(training_config["epochs"]),
        patience=int(training_config["patience"]),
        target_mean=target_mean,
        target_std=target_std,
    )
    model.load_state_dict(fit_result.best_state_dict)

    if workflow == "commercial_fit":
        assert commercial_contract is not None
        fit_report = {
            "status": "fit_completed",
            "workflow": workflow,
            "device": str(device),
            "seed": seed,
            "architecture": model_config["architecture"],
            "initialization": "random",
            "commercial_use_allowed": False,
            "promotion_status": "not_promoted",
            "best_epoch": fit_result.best_epoch,
            "best_validation_mae_kg": fit_result.best_validation_mae,
            "number_of_animals": {
                split: len({row["animal_id"] for row in split_rows[split]})
                for split in fit_splits
            },
            "number_of_images": {
                split: len(split_rows[split]) for split in fit_splits
            },
            "sampling": sampling_config or {"strategy": "uniform"},
            "history": fit_result.history,
            "held_out_partitions": ["calibration", "test"],
            "calibration_evaluated": False,
            "test_evaluated": False,
            "source_snapshot_id": commercial_contract.snapshot_id,
            "source_manifest_sha256": commercial_contract.manifest_sha256,
            "source_split_report_sha256": commercial_contract.split_report_sha256,
        }
        save_checkpoint(
            output_dir / "best_model.pt",
            state_dict=fit_result.best_state_dict,
            metadata={
                "architecture": model_config["architecture"],
                "dropout": float(model_config["dropout"]),
                "target_mean": target_mean,
                "target_std": target_std,
                "config": config,
                "epoch": fit_result.best_epoch,
                "workflow": workflow,
                "initialization": "random",
                "commercial_use_allowed": False,
                "promotion_status": "not_promoted",
                "held_out_partitions": ["calibration", "test"],
                "source_snapshot_id": commercial_contract.snapshot_id,
                "source_manifest_sha256": commercial_contract.manifest_sha256,
                "source_split_report_sha256": (
                    commercial_contract.split_report_sha256
                ),
            },
        )
        save_json(output_dir / "fit_metrics.json", fit_report)
        print(json.dumps(fit_report, indent=2, ensure_ascii=False))
        return

    test_result = evaluate_model(
        model,
        loaders["test"],
        device,
        target_mean=target_mean,
        target_std=target_std,
    )
    model_metrics = test_result.metrics
    baseline_metrics = regression_metrics(
        test_result.targets, [target_mean] * len(test_result.targets)
    )
    report = {
        "device": str(device),
        "seed": seed,
        "architecture": model_config["architecture"],
        "best_epoch": fit_result.best_epoch,
        "number_of_animals": {
            split: len({row["animal_id"] for row in split_rows[split]})
            for split in split_rows
        },
        "number_of_images": {split: len(split_rows[split]) for split in split_rows},
        "sampling": sampling_config or {"strategy": "uniform"},
        "model": model_metrics,
        "mean_baseline": baseline_metrics,
        "history": fit_result.history,
    }
    save_checkpoint(
        output_dir / "best_model.pt",
        state_dict=fit_result.best_state_dict,
        metadata={
            "architecture": model_config["architecture"],
            "dropout": float(model_config["dropout"]),
            "target_mean": target_mean,
            "target_std": target_std,
            "config": config,
            "epoch": fit_result.best_epoch,
        },
    )
    save_json(output_dir / "metrics.json", report)
    save_predictions(
        output_dir / "predictions_test.csv",
        split_rows["test"],
        targets=test_result.targets,
        predictions=test_result.predictions,
        indices=test_result.indices,
    )
    print(json.dumps(model_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
