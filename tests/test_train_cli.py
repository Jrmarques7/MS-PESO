from pathlib import Path

from ms_peso.train import parse_args


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
