from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ms_peso.depth_crop import clean_foreground_mask, largest_component_mask


@dataclass(frozen=True)
class ForegroundGeometry:
    height_m: float
    point_count: int
    mask_fraction: float
    box_height_fraction: float


def build_full_resolution_foreground_mask(
    depth: np.ndarray,
    background: np.ndarray,
    *,
    margin_mm: float = 150.0,
    max_depth_mm: float = 6000.0,
    downsample: int = 4,
    minimum_component_area: int = 30,
) -> np.ndarray:
    if depth.ndim != 2 or depth.shape != background.shape:
        raise ValueError("Profundidade e fundo devem ter a mesma forma bidimensional.")
    foreground = (
        (depth > 0)
        & (depth <= max_depth_mm)
        & (background > 0)
        & (depth + margin_mm < background)
    )
    cleaned = clean_foreground_mask(foreground, downsample=downsample)
    component = largest_component_mask(
        cleaned, minimum_area=minimum_component_area
    )
    expanded = np.repeat(
        np.repeat(component, downsample, axis=0), downsample, axis=1
    )
    full_mask = np.zeros(depth.shape, dtype=bool)
    height = min(depth.shape[0], expanded.shape[0])
    width = min(depth.shape[1], expanded.shape[1])
    full_mask[:height, :width] = expanded[:height, :width]
    return full_mask


def extract_foreground_geometry(
    depth: np.ndarray,
    background: np.ndarray,
    xyz: np.ndarray,
    *,
    margin_mm: float = 150.0,
    max_depth_mm: float = 6000.0,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    minimum_points: int = 100,
) -> ForegroundGeometry:
    if xyz.shape != (*depth.shape, 3):
        raise ValueError("XYZ deve corresponder pixel a pixel à profundidade.")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantis geométricos inválidos.")

    mask = build_full_resolution_foreground_mask(
        depth,
        background,
        margin_mm=margin_mm,
        max_depth_mm=max_depth_mm,
    )
    points = xyz[mask]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    if len(points) < minimum_points:
        raise ValueError(
            f"A geometria possui somente {len(points)} pontos válidos; "
            f"mínimo: {minimum_points}."
        )

    y_limits = np.quantile(points[:, 1], (lower_quantile, upper_quantile))
    height_m = float(y_limits[1] - y_limits[0])
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("A altura física extraída é inválida.")
    mask_rows = np.nonzero(mask)[0]
    box_height_fraction = float(
        (mask_rows.max() - mask_rows.min() + 1) / depth.shape[0]
    )
    return ForegroundGeometry(
        height_m=height_m,
        point_count=len(points),
        mask_fraction=float(mask.mean()),
        box_height_fraction=box_height_fraction,
    )
