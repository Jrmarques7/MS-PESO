from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean, pstdev

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ms_peso.artifacts import save_checkpoint, save_json, save_predictions
from ms_peso.config import load_yaml_config
from ms_peso.dataset import CattleWeightDataset
from ms_peso.evaluation import evaluate_model
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import read_manifest, validate_rows, write_manifest
from ms_peso.metrics import regression_metrics
from ms_peso.model import build_model
from ms_peso.reproducibility import set_global_seed
from ms_peso.self_supervised import load_self_supervised_encoder
from ms_peso.training import fit_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta encoder SSL usando somente treino e validação."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if config["project"].get("workflow") != "ssl_validation":
        raise ValueError("Configuração não pertence ao workflow ssl_validation.")
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    data_config = config["data"]
    manifest_path = Path(data_config["manifest"])
    rows = read_manifest(manifest_path)
    selected_view = data_config.get("view")
    if selected_view:
        rows = [row for row in rows if row.get("view") == selected_view]
    fit_rows = [row for row in rows if row.get("split") in {"train", "val"}]
    if {row.get("split") for row in fit_rows} != {"train", "val"}:
        raise ValueError("Treino e validação devem estar presentes.")
    if any(row.get("split") == "test" for row in fit_rows):
        raise ValueError("O ajuste SSL de validação não pode abrir o teste.")
    validate_rows(
        fit_rows,
        manifest_path=manifest_path,
        image_root=data_config["image_root"],
        check_images=True,
    )
    split_rows = {
        split: [row for row in fit_rows if row["split"] == split]
        for split in ("train", "val")
    }
    train_weights = [float(row["weight_kg"]) for row in split_rows["train"]]
    target_mean = fmean(train_weights)
    target_std = pstdev(train_weights)
    common = {
        "manifest_path": manifest_path,
        "image_root": data_config["image_root"],
        "image_size": int(data_config["image_size"]),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    datasets = {
        split: CattleWeightDataset(
            split_rows[split], training=split == "train", **common
        )
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data_config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }

    model_config = config["model"]
    if model_config.get("pretrained") is not False:
        raise ValueError("Ajuste SSL não aceita pesos ImageNet.")
    encoder = load_self_supervised_encoder(
        model_config["encoder_checkpoint"],
        expected_sha256=model_config["encoder_sha256"],
    )
    if encoder.architecture != model_config["architecture"]:
        raise ValueError("Arquitetura do encoder SSL diverge da configuração.")
    model = build_model(
        model_config["architecture"],
        pretrained=False,
        dropout=float(model_config["dropout"]),
    )
    incompatible = model.load_state_dict(encoder.state_dict, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
        "classifier.1.weight",
        "classifier.1.bias",
    }:
        raise ValueError(
            "Estado SSL incompatível com o regressor: "
            f"ausentes={incompatible.missing_keys}; "
            f"extras={incompatible.unexpected_keys}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit_result = fit_model(
        model,
        train_loader=loaders["train"],
        validation_loader=loaders["val"],
        optimizer=optimizer,
        criterion=nn.HuberLoss(delta=float(training["huber_delta"])),
        device=device,
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        target_mean=target_mean,
        target_std=target_std,
    )
    model.load_state_dict(fit_result.best_state_dict)
    validation = evaluate_model(
        model,
        loaders["val"],
        device,
        target_mean=target_mean,
        target_std=target_std,
    )
    mean_baseline = regression_metrics(
        validation.targets, [target_mean] * len(validation.targets)
    )
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "workflow": "ssl_validation",
        "device": str(device),
        "seed": seed,
        "architecture": model_config["architecture"],
        "pretrained": False,
        "initialization": "self_supervised_cc_by",
        "commercial_use_allowed": False,
        "promotion_status": "not_promoted",
        "best_epoch": fit_result.best_epoch,
        "number_of_animals": {
            split: len({row["animal_id"] for row in split_rows[split]})
            for split in ("train", "val")
        },
        "validation": validation.metrics,
        "mean_baseline": mean_baseline,
        "history": fit_result.history,
        "encoder_sha256": encoder.sha256,
        "encoder_source_manifest_sha256": encoder.source_manifest_sha256,
        "supervised_manifest_sha256": calculate_sha256(manifest_path),
        "validation_used_for_selection": True,
        "test_evaluated": False,
    }
    save_checkpoint(
        output_dir / "best_validation_model.pt",
        state_dict=fit_result.best_state_dict,
        metadata={
            **{key: value for key, value in report.items() if key != "history"},
            "dropout": float(model_config["dropout"]),
            "target_mean": target_mean,
            "target_std": target_std,
            "config": config,
        },
    )
    save_json(output_dir / "validation_metrics.json", report)
    write_manifest(fit_rows, output_dir / "resolved_fit_manifest.csv")
    save_predictions(
        output_dir / "predictions_validation.csv",
        split_rows["val"],
        targets=validation.targets,
        predictions=validation.predictions,
        indices=validation.indices,
    )
    print(f"Validação SSL MAE: {validation.metrics['mae_kg']:.2f} kg")
    print("Teste preservado: nenhuma linha de teste foi aberta ou avaliada.")


if __name__ == "__main__":
    main()
