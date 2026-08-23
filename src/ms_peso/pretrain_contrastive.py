from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ms_peso.artifacts import save_checkpoint, save_json
from ms_peso.config import load_yaml_config
from ms_peso.integrity import calculate_sha256
from ms_peso.prepare_ssl_manifest import read_ssl_manifest
from ms_peso.reproducibility import set_global_seed
from ms_peso.self_supervised import (
    ContrastiveEfficientNet,
    ContrastiveImageDataset,
    nt_xent_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pré-treina um encoder bovino sem rótulos nem ImageNet."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if config["model"].get("pretrained") is not False:
        raise ValueError("Pré-treinamento CC BY exige model.pretrained=false.")
    if config["model"].get("initialization") != "random":
        raise ValueError("Pré-treinamento CC BY exige initialization=random.")

    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    data_config = config["data"]
    manifest_path = Path(data_config["manifest"])
    rows = read_ssl_manifest(manifest_path)
    if {row["source_license"] for row in rows} != {"CC_BY_4_0"}:
        raise ValueError("Todas as imagens SSL devem possuir licença CC BY 4.0.")
    image_root = Path(data_config["image_root"])
    for row in rows:
        image_path = image_root / row["image_path"]
        if calculate_sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Hash da imagem SSL divergiu: {image_path}")

    dataset = ContrastiveImageDataset(
        rows,
        image_root=image_root,
        image_size=int(data_config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveEfficientNet(
        projection_dim=int(config["model"]["projection_dim"])
    ).to(device)
    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for first, second in loader:
            first = first.to(device)
            second = second.to(device)
            optimizer.zero_grad(set_to_none=True)
            first_embedding = model(first)
            second_embedding = model(second)
            loss = nt_xent_loss(
                first_embedding,
                second_embedding,
                temperature=float(training["temperature"]),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * first.shape[0]
            samples += first.shape[0]
        epoch_loss = total_loss / samples
        history.append({"epoch": epoch, "contrastive_loss": epoch_loss})
        print(f"época SSL {epoch:03d} | loss={epoch_loss:.4f}")

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_counts = Counter(row["source_dataset"] for row in rows)
    metadata = {
        "checkpoint_type": "self_supervised_encoder",
        "architecture": "efficientnet_b0",
        "pretrained": False,
        "initialization": "random",
        "labels_used": False,
        "commercial_use_allowed": False,
        "promotion_status": "not_promoted",
        "source_license": "CC_BY_4_0",
        "source_manifest_sha256": calculate_sha256(manifest_path),
        "source_counts": dict(source_counts),
        "config": config,
    }
    save_checkpoint(
        output_dir / "encoder_final.pt",
        state_dict=model.encoder.state_dict(),
        metadata=metadata,
    )
    save_json(
        output_dir / "pretrain_metrics.json",
        {
            **metadata,
            "device": str(device),
            "seed": seed,
            "number_of_images": len(rows),
            "history": history,
            "test_evaluated": False,
        },
    )
    print(f"Encoder SSL gravado em {output_dir / 'encoder_final.pt'}")


if __name__ == "__main__":
    main()
