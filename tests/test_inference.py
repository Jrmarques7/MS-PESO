from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from ms_peso.inference import CattleWeightPredictor, resolve_device
from ms_peso.model_package import ModelDescriptor, calculate_sha256


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.5]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(inputs.shape[0], 1)


def make_descriptor(tmp_path: Path, checkpoint_path: Path) -> ModelDescriptor:
    model_card = tmp_path / "model_card.md"
    model_card.write_text("# Model card", encoding="utf-8")
    return ModelDescriptor(
        model_id="test-model",
        model_version="1",
        status="experimental",
        production_ready=False,
        architecture="efficientnet_b0",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=calculate_sha256(checkpoint_path),
        image_size=16,
        input_view="left",
        dataset="test",
        breed="hereford",
        limitations=("Apenas teste.",),
        model_card_path=model_card,
    )


def test_loads_checkpoint_and_predicts_weight(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": TinyRegressor().state_dict(),
            "architecture": "efficientnet_b0",
            "dropout": 0.2,
            "target_mean": 400.0,
            "target_std": 20.0,
            "config": {"data": {"image_size": 16}},
        },
        checkpoint_path,
    )
    descriptor = make_descriptor(tmp_path, checkpoint_path)
    monkeypatch.setattr(
        "ms_peso.inference.build_model", lambda *args, **kwargs: TinyRegressor()
    )
    image_path = tmp_path / "cow.png"
    Image.new("RGB", (32, 24), color=(100, 120, 140)).save(image_path)

    predictor = CattleWeightPredictor.load(descriptor, device="cpu")
    prediction = predictor.predict_image(image_path)

    assert prediction.estimated_weight_kg == pytest.approx(410)
    assert prediction.original_width == 32
    assert prediction.original_height == 24
    assert str(predictor.device) == "cpu"


def test_rejects_invalid_image(tmp_path: Path) -> None:
    invalid_image = tmp_path / "invalid.png"
    invalid_image.write_text("not an image", encoding="utf-8")
    predictor = CattleWeightPredictor(
        descriptor=ModelDescriptor(
            model_id="test",
            model_version="1",
            status="experimental",
            production_ready=False,
            architecture="efficientnet_b0",
            checkpoint_path=tmp_path / "unused.pt",
            checkpoint_sha256="0" * 64,
            image_size=16,
            input_view="left",
            dataset="test",
            breed="test",
            limitations=("test",),
            model_card_path=tmp_path / "unused.md",
        ),
        model=TinyRegressor(),
        device=torch.device("cpu"),
        target_mean=400,
        target_std=20,
    )

    with pytest.raises(ValueError, match="imagem inválido"):
        predictor.predict_image(invalid_image)


def test_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="auto, cpu ou cuda"):
        resolve_device("tpu")
