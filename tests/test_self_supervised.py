from pathlib import Path

import pytest
import torch
from PIL import Image

from ms_peso.self_supervised import (
    ContrastiveImageDataset,
    nt_xent_loss,
)


def test_contrastive_dataset_returns_two_augmented_views(tmp_path: Path) -> None:
    Image.new("RGB", (48, 32), color=(20, 30, 40)).save(tmp_path / "cow.png")
    dataset = ContrastiveImageDataset(
        [{"image_path": "cow.png"}], image_root=tmp_path, image_size=24
    )
    first, second = dataset[0]
    assert first.shape == second.shape == (3, 24, 24)


def test_nt_xent_is_finite_and_differentiable() -> None:
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    loss = nt_xent_loss(first, second, temperature=0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert first.grad is not None
    assert second.grad is not None


def test_nt_xent_rejects_single_sample_batch() -> None:
    with pytest.raises(ValueError, match="ao menos duas"):
        nt_xent_loss(torch.ones(1, 4), torch.ones(1, 4), temperature=0.2)
