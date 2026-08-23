from pathlib import Path

import pytest
from PIL import Image

from ms_peso.image_quality import (
    ImageQualityPolicy,
    assess_image_quality,
    load_image_quality_policy,
)


def policy() -> ImageQualityPolicy:
    return ImageQualityPolicy(
        policy_id="test-policy",
        policy_version="1",
        analysis_max_dimension=512,
        min_width=480,
        min_height=320,
        min_aspect_ratio=1.15,
        max_aspect_ratio=2.0,
        min_mean_luma=35,
        max_mean_luma=220,
        max_dark_fraction=0.8,
        max_bright_fraction=0.8,
        min_sharpness=100,
        limitations=("Não valida pose.",),
    )


def checkerboard(path: Path, *, width: int = 800, height: int = 500) -> None:
    image = Image.new("L", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = 32 if (x // 16 + y // 16) % 2 else 224
    image.convert("RGB").save(path)


def test_accepts_sharp_landscape_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sharp.png"
    checkerboard(image_path)

    report = assess_image_quality(image_path, policy())

    assert report.accepted is True
    assert report.status == "passed"
    assert report.rejection_reasons == ()
    assert all(check.passed for check in report.checks)


def test_rejects_small_dark_portrait_and_blurred_image(tmp_path: Path) -> None:
    image_path = tmp_path / "invalid.png"
    Image.new("RGB", (200, 400), (0, 0, 0)).save(image_path)

    report = assess_image_quality(image_path, policy())
    failed_codes = {check.code for check in report.checks if not check.passed}

    assert report.accepted is False
    assert report.status == "rejected"
    assert {
        "minimum_resolution",
        "landscape_aspect_ratio",
        "mean_luma",
        "dark_clipping",
        "sharpness",
    } <= failed_codes
    assert len(report.rejection_reasons) == len(failed_codes)


def test_rejects_invalid_policy_interval(tmp_path: Path) -> None:
    policy_path = tmp_path / "quality.yaml"
    policy_path.write_text(
        """
schema_version: 1
policy_id: invalid
policy_version: "1"
analysis:
  max_dimension: 512
thresholds:
  min_width: 480
  min_height: 320
  min_aspect_ratio: 2.0
  max_aspect_ratio: 1.0
  min_mean_luma: 35
  max_mean_luma: 220
  max_dark_fraction: 0.8
  max_bright_fraction: 0.8
  min_sharpness: 100
limitations:
  - Teste.
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="proporção"):
        load_image_quality_policy(policy_path)
