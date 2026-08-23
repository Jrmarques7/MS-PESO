import pytest
import torch

from ms_peso.model import build_model


@pytest.mark.parametrize(
    ("architecture", "channels"),
    [
        ("resnet18", 3),
        ("efficientnet_b0", 3),
        ("convnext_tiny", 3),
        ("efficientnet_b0_rgb_depth", 4),
        ("efficientnet_b0_multiview", 6),
    ],
)
def test_model_outputs_one_value_per_image(architecture: str, channels: int) -> None:
    model = build_model(architecture, pretrained=False)

    output = model(torch.zeros(2, channels, 64, 64))

    assert output.shape == (2, 1)


def test_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="não implementada"):
        build_model("unknown", pretrained=False)


def test_rgb_depth_model_rejects_wrong_channel_count() -> None:
    model = build_model("efficientnet_b0_rgb_depth", pretrained=False)

    with pytest.raises(ValueError, match="N×4×H×W"):
        model(torch.zeros(2, 3, 64, 64))


def test_multi_view_model_rejects_wrong_channel_count() -> None:
    model = build_model("efficientnet_b0_multiview", pretrained=False)

    with pytest.raises(ValueError, match="N×6×H×W"):
        model(torch.zeros(2, 3, 64, 64))
