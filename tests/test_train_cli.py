from pathlib import Path

import pytest

from ms_peso.train import parse_args, resolve_model_initialization


def test_training_cli_accepts_reproducibility_overrides() -> None:
    result = parse_args(
        [
            "--config",
            "config.yaml",
            "--seed",
            "43",
            "--output-dir",
            "artifacts/repeat",
        ]
    )

    assert result.config == "config.yaml"
    assert result.seed == 43
    assert result.output_dir == Path("artifacts/repeat")


def test_random_initialization_requires_pretrained_false() -> None:
    assert (
        resolve_model_initialization(
            {"pretrained": False, "initialization": "random"}
        )
        == "random"
    )
    with pytest.raises(ValueError, match="incompatível"):
        resolve_model_initialization(
            {"pretrained": True, "initialization": "random"}
        )


def test_non_pretrained_model_rejects_non_random_initialization() -> None:
    with pytest.raises(ValueError, match="exige model.initialization=random"):
        resolve_model_initialization(
            {"pretrained": False, "initialization": "imagenet"}
        )
