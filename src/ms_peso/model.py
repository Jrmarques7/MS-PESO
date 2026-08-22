from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
    architecture: str = "resnet18",
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(
            f"Arquitetura ainda não implementada: {architecture}. Use resnet18."
        )
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, 1))
    return model
