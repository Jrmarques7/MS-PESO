from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)


class RGBDepthEfficientNet(nn.Module):
    """Funde representações globais sem presumir registro pixel a pixel."""

    def __init__(self, *, pretrained: bool, dropout: float) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.rgb_encoder = efficientnet_b0(weights=weights)
        rgb_features = self.rgb_encoder.classifier[1].in_features
        self.rgb_encoder.classifier = nn.Identity()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(rgb_features + 128, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("RGBDepthEfficientNet espera tensores N×4×H×W.")
        rgb_features = self.rgb_encoder(inputs[:, :3])
        depth_features = self.depth_encoder(inputs[:, 3:4])
        return self.regressor(torch.cat((rgb_features, depth_features), dim=1))


class MultiViewEfficientNet(nn.Module):
    """Codifica duas vistas RGB com o mesmo EfficientNet e funde seus vetores."""

    def __init__(self, *, pretrained: bool, dropout: float) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.encoder = efficientnet_b0(weights=weights)
        number_of_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(number_of_features * 2, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 6:
            raise ValueError("MultiViewEfficientNet espera tensores N×6×H×W.")
        batch_size = inputs.shape[0]
        features = self.encoder(
            torch.cat((inputs[:, :3], inputs[:, 3:6]), dim=0)
        )
        primary_features = features[:batch_size]
        secondary_features = features[batch_size:]
        combined_features = torch.cat((primary_features, secondary_features), dim=1)
        return self.regressor(combined_features)


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
    if architecture == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        input_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, 1)
        )
        return model
    if architecture == "efficientnet_b0_rgb_depth":
        return RGBDepthEfficientNet(pretrained=pretrained, dropout=dropout)
    if architecture == "efficientnet_b0_multiview":
        return MultiViewEfficientNet(pretrained=pretrained, dropout=dropout)
    raise ValueError(
        f"Arquitetura não implementada: {architecture}. "
        "Use resnet18, efficientnet_b0, convnext_tiny ou "
        "efficientnet_b0_rgb_depth ou efficientnet_b0_multiview."
    )
