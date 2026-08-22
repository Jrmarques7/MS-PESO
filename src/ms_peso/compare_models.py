from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnimalPrediction:
    animal_id: str
    actual_kg: float
    predicted_kg: float


Metric = Callable[[Sequence[AnimalPrediction]], float]


def load_animal_predictions(path: str | Path) -> dict[str, AnimalPrediction]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("O arquivo de predições está vazio")
    predictions = {}
    for row in rows:
        animal_id = row["animal_id"]
        if animal_id in predictions:
            raise ValueError(f"animal_id duplicado: {animal_id}")
        predictions[animal_id] = AnimalPrediction(
            animal_id=animal_id,
            actual_kg=float(row["weight_kg"]),
            predicted_kg=float(row["predicted_weight_kg"]),
        )
    return predictions


def mae(items: Sequence[AnimalPrediction]) -> float:
    return sum(abs(item.predicted_kg - item.actual_kg) for item in items) / len(items)


def rmse(items: Sequence[AnimalPrediction]) -> float:
    return math.sqrt(
        sum((item.predicted_kg - item.actual_kg) ** 2 for item in items) / len(items)
    )


def mape(items: Sequence[AnimalPrediction]) -> float:
    return (
        100
        * sum(
            abs(item.predicted_kg - item.actual_kg) / item.actual_kg for item in items
        )
        / len(items)
    )


def bias(items: Sequence[AnimalPrediction]) -> float:
    return sum(item.predicted_kg - item.actual_kg for item in items) / len(items)


METRICS: dict[str, Metric] = {
    "mae_kg": mae,
    "rmse_kg": rmse,
    "mape_pct": mape,
    "bias_kg": bias,
}


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _interval(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "lower_95": percentile(ordered, 0.025),
        "upper_95": percentile(ordered, 0.975),
    }


def compare(
    reference: dict[str, AnimalPrediction],
    candidate: dict[str, AnimalPrediction],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    if iterations < 100:
        raise ValueError("iterations deve ser pelo menos 100")
    if reference.keys() != candidate.keys():
        raise ValueError("Os modelos devem conter os mesmos animal_id")
    animal_ids = sorted(reference)
    for animal_id in animal_ids:
        if not math.isclose(
            reference[animal_id].actual_kg,
            candidate[animal_id].actual_kg,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Peso real divergente para {animal_id}")

    reference_items = [reference[animal_id] for animal_id in animal_ids]
    candidate_items = [candidate[animal_id] for animal_id in animal_ids]
    distributions = {
        metric_name: {"reference": [], "candidate": [], "delta": []}
        for metric_name in METRICS
    }
    rng = random.Random(seed)
    for _ in range(iterations):
        indices = [rng.randrange(len(animal_ids)) for _ in animal_ids]
        reference_sample = [reference_items[index] for index in indices]
        candidate_sample = [candidate_items[index] for index in indices]
        for metric_name, metric in METRICS.items():
            reference_value = metric(reference_sample)
            candidate_value = metric(candidate_sample)
            distributions[metric_name]["reference"].append(reference_value)
            distributions[metric_name]["candidate"].append(candidate_value)
            distributions[metric_name]["delta"].append(
                candidate_value - reference_value
            )

    result: dict[str, object] = {
        "animals": len(animal_ids),
        "iterations": iterations,
        "seed": seed,
        "metrics": {},
    }
    metrics_result = result["metrics"]
    assert isinstance(metrics_result, dict)
    for metric_name, metric in METRICS.items():
        delta_values = distributions[metric_name]["delta"]
        metrics_result[metric_name] = {
            "reference": {
                "point": metric(reference_items),
                **_interval(distributions[metric_name]["reference"]),
            },
            "candidate": {
                "point": metric(candidate_items),
                **_interval(distributions[metric_name]["candidate"]),
            },
            "delta_candidate_minus_reference": {
                "point": metric(candidate_items) - metric(reference_items),
                **_interval(delta_values),
            },
            "probability_candidate_lower": sum(value < 0 for value in delta_values)
            / iterations,
        }
    return result


def markdown_report(result: dict[str, object], reference: str, candidate: str) -> str:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    lines = [
        f"# Comparação pareada — {candidate} × {reference}",
        "",
        f"Bootstrap pareado por {result['animals']} animais, "
        f"{result['iterations']} iterações e seed {result['seed']}.",
        "",
        "| Métrica | Referência (IC95%) | Candidato (IC95%) | "
        "Diferença (IC95%) | P(candidato menor) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("mae_kg", "rmse_kg", "mape_pct", "bias_kg"):
        values = metrics[name]
        reference_values = values["reference"]
        candidate_values = values["candidate"]
        delta = values["delta_candidate_minus_reference"]
        lines.append(
            f"| {name} | {reference_values['point']:.2f} "
            f"[{reference_values['lower_95']:.2f}, "
            f"{reference_values['upper_95']:.2f}] | "
            f"{candidate_values['point']:.2f} "
            f"[{candidate_values['lower_95']:.2f}, "
            f"{candidate_values['upper_95']:.2f}] | "
            f"{delta['point']:+.2f} [{delta['lower_95']:+.2f}, "
            f"{delta['upper_95']:+.2f}] | "
            f"{100 * values['probability_candidate_lower']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "A diferença só é considerada sustentada por este teste quando o intervalo "
            "pareado não cruza zero. O conjunto pequeno produz intervalos largos e não "
            "substitui validação externa.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara dois modelos por animal")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare(
        load_animal_predictions(args.reference),
        load_animal_predictions(args.candidate),
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(
        markdown_report(result, args.reference_label, args.candidate_label),
        encoding="utf-8",
    )
    print(f"Comparação gravada em {args.output_dir}")


if __name__ == "__main__":
    main()
