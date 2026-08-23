from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as torch_functional

from ms_peso.manifest import resolve_manifest_path


@dataclass(frozen=True)
class RelativeBox:
    left: float
    top: float
    right: float
    bottom: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            max(0, round(self.left * width)),
            max(0, round(self.top * height)),
            min(width, round(self.right * width)),
            min(height, round(self.bottom * height)),
        )


def load_depth_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Profundidade deve ter um canal: {path}")
    return depth


def estimate_background_depth(
    depth_images: list[np.ndarray], *, percentile: float = 100.0
) -> np.ndarray:
    if not depth_images:
        raise ValueError("Ao menos uma profundidade de treino é necessária.")
    shape = depth_images[0].shape
    if any(image.shape != shape for image in depth_images):
        raise ValueError("Todas as profundidades devem ter a mesma resolução.")
    stack = np.stack(depth_images).astype(np.float32, copy=False)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        background = np.nanpercentile(stack, percentile, axis=0)
    return np.nan_to_num(background, nan=0.0).astype(np.float32)


def build_training_background(
    rows: list[dict[str, str]],
    *,
    manifest_path: str | Path,
    image_root: str | Path | None,
    depth_image_column: str,
    percentile: float = 100.0,
) -> np.ndarray:
    training_rows = [row for row in rows if row.get("split") == "train"]
    if not training_rows:
        raise ValueError("O fundo deve ser estimado somente com linhas de treino.")
    images = [
        load_depth_image(
            resolve_manifest_path(
                row, depth_image_column, manifest_path, image_root
            )
        )
        for row in training_rows
    ]
    return estimate_background_depth(images, percentile=percentile)


def _erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    return 1 - torch_functional.max_pool2d(
        1 - mask, kernel_size=kernel_size, stride=1, padding=padding
    )


def clean_foreground_mask(mask: np.ndarray, *, downsample: int = 4) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("A máscara de profundidade deve ser bidimensional.")
    height = mask.shape[0] // downsample * downsample
    width = mask.shape[1] // downsample * downsample
    tensor = torch.from_numpy(mask[:height, :width].astype(np.float32))[None, None]
    tensor = torch_functional.avg_pool2d(
        tensor, kernel_size=downsample, stride=downsample
    )
    tensor = (tensor >= 0.20).to(torch.float32)
    tensor = torch_functional.max_pool2d(_erode(tensor, 3), 3, stride=1, padding=1)
    tensor = _erode(
        torch_functional.max_pool2d(tensor, 5, stride=1, padding=2), 5
    )
    return tensor[0, 0].numpy().astype(bool)


def largest_component_mask(
    mask: np.ndarray, *, minimum_area: int = 30
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("A máscara deve ser bidimensional.")
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels
    if len(best_pixels) < minimum_area:
        raise ValueError("Nenhum componente de primeiro plano confiável encontrado.")
    component = np.zeros_like(mask, dtype=bool)
    ys, xs = zip(*best_pixels, strict=True)
    component[np.asarray(ys), np.asarray(xs)] = True
    return component


def largest_component_box(mask: np.ndarray, *, minimum_area: int = 30) -> RelativeBox:
    component = largest_component_mask(mask, minimum_area=minimum_area)
    height, width = component.shape
    ys, xs = np.nonzero(component)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return RelativeBox(left / width, top / height, right / width, bottom / height)


def add_box_padding(box: RelativeBox, padding: float) -> RelativeBox:
    return RelativeBox(
        max(0.0, box.left - padding),
        max(0.0, box.top - padding),
        min(1.0, box.right + padding),
        min(1.0, box.bottom + padding),
    )


def detect_depth_foreground_box(
    depth: np.ndarray,
    background: np.ndarray,
    *,
    margin_mm: float = 150.0,
    max_depth_mm: float = 6000.0,
    padding: float = 0.08,
) -> RelativeBox:
    if depth.shape != background.shape:
        raise ValueError("Profundidade e fundo devem ter a mesma resolução.")
    foreground = (
        (depth > 0)
        & (depth <= max_depth_mm)
        & (background > 0)
        & (depth + margin_mm < background)
    )
    cleaned = clean_foreground_mask(foreground)
    return add_box_padding(largest_component_box(cleaned), padding)


def render_depth_guided_rgb(
    image: Image.Image,
    box: RelativeBox,
    *,
    output_mode: str,
    fill_color: tuple[int, int, int] = (124, 116, 104),
) -> Image.Image:
    rgb_image = image.convert("RGB")
    pixel_box = box.to_pixels(*rgb_image.size)
    if output_mode == "crop":
        return rgb_image.crop(pixel_box)
    if output_mode == "masked_canvas":
        output = Image.new("RGB", rgb_image.size, color=fill_color)
        output.paste(rgb_image.crop(pixel_box), pixel_box[:2])
        return output
    raise ValueError(f"Modo de saída desconhecido: {output_mode}")
