from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ms_peso.evaluation import evaluate_model


@dataclass(frozen=True)
class FitResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_mae: float
    history: list[dict[str, float | int]]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Atualiza o modelo por uma única época e retorna a perda média."""
    model.train()
    total_loss = 0.0
    for images, normalized_targets, _ in loader:
        images = images.to(device)
        normalized_targets = normalized_targets.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images).squeeze(1)
        loss = criterion(predictions, normalized_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def fit_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    target_mean: float,
    target_std: float,
) -> FitResult:
    """Treina com early stopping, sem decidir como artefatos serão persistidos."""
    best_mae = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = evaluate_model(
            model,
            validation_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )
        validation_metrics = validation.metrics
        history.append({"epoch": epoch, "train_loss": train_loss, **validation_metrics})
        print(
            f"época {epoch:03d} | loss={train_loss:.4f} | "
            f"val_MAE={validation_metrics['mae_kg']:.2f} kg"
        )

        if validation_metrics["mae_kg"] < best_mae:
            best_mae = validation_metrics["mae_kg"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping acionado.")
                break

    if not best_state_dict:
        raise RuntimeError("O treinamento terminou sem produzir um estado válido.")
    return FitResult(best_state_dict, best_epoch, best_mae, history)
