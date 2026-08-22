from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


def build_model(
    architecture: str = "resnet18",
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, 1))
        return model
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, 1)
        )
        return model
    raise ValueError(
        f"Arquitetura não implementada: {architecture}. "
        "Use resnet18 ou efficientnet_b0."
    )
