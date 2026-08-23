import numpy as np
import pytest
from PIL import Image

from ms_peso.depth_crop import (
    RelativeBox,
    detect_depth_foreground_box,
    estimate_background_depth,
    largest_component_box,
    render_depth_guided_rgb,
)


def test_estimates_far_background_while_ignoring_invalid_depth() -> None:
    images = [
        np.array([[0, 2000], [3000, 4000]], dtype=np.float32),
        np.array([[0, 5000], [3500, 4500]], dtype=np.float32),
    ]

    background = estimate_background_depth(images, percentile=100)

    assert background.tolist() == [[0, 5000], [3500, 4500]]


def test_finds_largest_component_box() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:4, 2:4] = True
    mask[6:16, 5:18] = True

    box = largest_component_box(mask, minimum_area=10)

    assert box.left == pytest.approx(5 / 20)
    assert box.top == pytest.approx(6 / 20)
    assert box.right == pytest.approx(18 / 20)
    assert box.bottom == pytest.approx(16 / 20)


def test_detects_closer_rectangle_against_background() -> None:
    background = np.full((80, 80), 5000, dtype=np.float32)
    depth = background.copy()
    depth[20:60, 16:68] = 2000

    box = detect_depth_foreground_box(depth, background, padding=0)

    assert box.left == pytest.approx(16 / 80, abs=0.06)
    assert box.top == pytest.approx(20 / 80, abs=0.06)
    assert box.right == pytest.approx(68 / 80, abs=0.06)
    assert box.bottom == pytest.approx(60 / 80, abs=0.06)


def test_rejects_scene_without_foreground() -> None:
    background = np.full((80, 80), 5000, dtype=np.float32)

    with pytest.raises(ValueError, match="Nenhum componente"):
        detect_depth_foreground_box(background.copy(), background)


def test_masks_outside_box_without_changing_canvas_size() -> None:
    image = Image.new("RGB", (100, 50), color=(255, 0, 0))
    box = RelativeBox(0.25, 0.20, 0.75, 0.80)

    output = render_depth_guided_rgb(image, box, output_mode="masked_canvas")

    assert output.size == image.size
    assert output.getpixel((50, 25)) == (255, 0, 0)
    assert output.getpixel((5, 5)) == (124, 116, 104)


def test_crops_to_box() -> None:
    image = Image.new("RGB", (100, 50), color=(255, 0, 0))
    box = RelativeBox(0.25, 0.20, 0.75, 0.80)

    output = render_depth_guided_rgb(image, box, output_mode="crop")

    assert output.size == (50, 30)
