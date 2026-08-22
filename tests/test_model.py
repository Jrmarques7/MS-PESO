import pytest
import torch

from ms_peso.model import build_model


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_model_outputs_one_value_per_image(architecture: str) -> None:
    model = build_model(architecture, pretrained=False)

    output = model(torch.zeros(2, 3, 64, 64))

    assert output.shape == (2, 1)


def test_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="não implementada"):
        build_model("unknown", pretrained=False)
