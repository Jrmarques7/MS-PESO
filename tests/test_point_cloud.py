from pathlib import Path

import numpy as np
import pytest

from ms_peso.point_cloud import read_organized_ply_xyz


def write_test_ply(path: Path, points: np.ndarray, *, file_format: str) -> None:
    header = (
        "ply\n"
        f"format {file_format} 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1")])
    vertices = np.zeros(len(points), dtype=dtype)
    for coordinate in ("x", "y", "z"):
        vertices[coordinate] = points[:, "xyz".index(coordinate)]
    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)


def test_reads_binary_organized_xyz(tmp_path: Path) -> None:
    points = np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        dtype=np.float32,
    )
    path = tmp_path / "cloud.ply"
    write_test_ply(path, points, file_format="binary_little_endian")

    result = read_organized_ply_xyz(path, width=2, height=2)

    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result.reshape(-1, 3), points)


def test_rejects_wrong_vertex_count(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    write_test_ply(path, np.zeros((3, 3)), file_format="binary_little_endian")

    with pytest.raises(ValueError, match="esperado"):
        read_organized_ply_xyz(path, width=2, height=2)


def test_rejects_ascii_ply(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    write_test_ply(path, np.zeros((4, 3)), file_format="ascii")

    with pytest.raises(ValueError, match="little-endian"):
        read_organized_ply_xyz(path, width=2, height=2)
