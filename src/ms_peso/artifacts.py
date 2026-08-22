from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def save_checkpoint(
    path: str | Path,
    *,
    state_dict: Mapping[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Persiste exclusivamente um checkpoint e seus metadados."""
    import torch

    torch.save({"model_state_dict": state_dict, **metadata}, Path(path))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste exclusivamente um relatório JSON."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=True)


def save_predictions(
    path: str | Path,
    rows: list[dict[str, str]],
    *,
    targets: Sequence[float],
    predictions: Sequence[float],
    indices: Sequence[int],
) -> None:
    """Persiste exclusivamente as predições de uma avaliação."""
    if not (len(targets) == len(predictions) == len(indices)):
        raise ValueError("targets, predictions e indices devem ter o mesmo tamanho")
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "animal_id",
                "event_id",
                "weight_kg",
                "predicted_weight_kg",
                "error_kg",
            ],
        )
        writer.writeheader()
        for index, target, prediction in zip(
            indices,
            targets,
            predictions,
            strict=True,
        ):
            row = rows[index]
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "animal_id": row["animal_id"],
                    "event_id": row["event_id"],
                    "weight_kg": f"{target:.4f}",
                    "predicted_weight_kg": f"{prediction:.4f}",
                    "error_kg": f"{prediction - target:.4f}",
                }
            )
