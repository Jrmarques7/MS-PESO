from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def weight_band_index(weight_kg: float, boundaries: Sequence[float]) -> int:
    """Mapeia exclusivamente um peso para uma faixa ordenada."""
    return sum(weight_kg >= boundary for boundary in boundaries)


def inverse_frequency_weights(
    weights_kg: Sequence[float],
    *,
    boundaries: Sequence[float],
    power: float = 0.5,
) -> list[float]:
    """Calcula pesos de amostragem por faixa sem construir o DataLoader."""
    if not weights_kg:
        raise ValueError("weights_kg não pode ser vazio")
    if sorted(boundaries) != list(boundaries) or len(set(boundaries)) != len(
        boundaries
    ):
        raise ValueError("boundaries deve ser estritamente crescente")
    if not 0 < power <= 1:
        raise ValueError("power deve estar no intervalo (0, 1]")
    bands = [weight_band_index(weight, boundaries) for weight in weights_kg]
    counts = Counter(bands)
    return [(1 / counts[band]) ** power for band in bands]
