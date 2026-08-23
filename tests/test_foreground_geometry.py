import numpy as np
import pytest

from ms_peso.foreground_geometry import extract_foreground_geometry


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 80, 80
    background = np.full((height, width), 5000, dtype=np.float32)
    depth = background.copy()
    depth[20:60, 16:68] = 2000
    x = np.broadcast_to(np.linspace(-1, 1, width), (height, width))
    y = np.broadcast_to(np.linspace(-1, 1, height)[:, None], (height, width))
    z = depth / 1000
    xyz = np.stack((x, y, z), axis=2).astype(np.float32)
    return depth, background, xyz


def test_extracts_robust_physical_height() -> None:
    depth, background, xyz = synthetic_inputs()

    geometry = extract_foreground_geometry(depth, background, xyz)

    assert 0.7 < geometry.height_m < 1.1
    assert geometry.point_count > 1000
    assert 0.2 < geometry.mask_fraction < 0.5
    assert 0.4 < geometry.box_height_fraction < 0.7


def test_rejects_unregistered_point_cloud() -> None:
    depth, background, xyz = synthetic_inputs()

    with pytest.raises(ValueError, match="pixel a pixel"):
        extract_foreground_geometry(depth, background, xyz[:-1])


def test_rejects_too_few_valid_points() -> None:
    depth, background, xyz = synthetic_inputs()
    xyz[:] = np.nan

    with pytest.raises(ValueError, match="pontos válidos"):
        extract_foreground_geometry(depth, background, xyz)
