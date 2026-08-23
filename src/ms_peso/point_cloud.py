from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_SCALAR_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def _read_vertex_layout(file) -> tuple[int, np.dtype]:
    if file.readline().strip() != b"ply":
        raise ValueError("O arquivo não possui cabeçalho PLY.")

    file_format = ""
    current_element = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    while True:
        raw_line = file.readline()
        if not raw_line:
            raise ValueError("Cabeçalho PLY incompleto.")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("Cabeçalho PLY inválido.") from exc
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            file_format = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("Propriedade PLY de vértice não suportada.")
            scalar_type = PLY_SCALAR_TYPES.get(parts[1])
            if scalar_type is None:
                raise ValueError(f"Tipo escalar PLY não suportado: {parts[1]}")
            vertex_properties.append((parts[2], scalar_type))
        elif parts[0] == "end_header":
            break

    if file_format != "binary_little_endian":
        raise ValueError("Somente PLY binário little-endian é suportado.")
    if vertex_count <= 0 or not vertex_properties:
        raise ValueError("O PLY não contém vértices válidos.")
    property_names = {name for name, _ in vertex_properties}
    if not {"x", "y", "z"}.issubset(property_names):
        raise ValueError("O PLY precisa conter as coordenadas x, y e z.")
    return vertex_count, np.dtype(vertex_properties)


def read_organized_ply_xyz(
    path: str | Path,
    *,
    width: int = 512,
    height: int = 424,
) -> np.ndarray:
    """Lê coordenadas XYZ de uma nuvem PCL organizada, preservando os pixels."""
    point_cloud_path = Path(path)
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"Nuvem de pontos não encontrada: {point_cloud_path}")
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura da nuvem devem ser positivas.")

    with point_cloud_path.open("rb") as file:
        vertex_count, vertex_dtype = _read_vertex_layout(file)
        if vertex_count != width * height:
            raise ValueError(
                f"PLY possui {vertex_count} vértices; esperado: {width * height}."
            )
        vertices = np.fromfile(file, dtype=vertex_dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError("Dados binários PLY estão truncados.")

    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    return xyz.astype(np.float32, copy=False).reshape(height, width, 3)
