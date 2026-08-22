from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    animal_id: str
    actual_kg: float
    predicted_kg: float

    @property
    def error_kg(self) -> float:
        return self.predicted_kg - self.actual_kg

    @property
    def absolute_error_kg(self) -> float:
        return abs(self.error_kg)


WEIGHT_BANDS = (
    ("<350 kg", -math.inf, 350),
    ("350–399 kg", 350, 400),
    ("400–449 kg", 400, 450),
    ("450–499 kg", 450, 500),
    ("≥500 kg", 500, math.inf),
)


def load_predictions(path: str | Path) -> list[Prediction]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("O arquivo de predições está vazio")
    return [
        Prediction(
            row["animal_id"], float(row["weight_kg"]), float(row["predicted_weight_kg"])
        )
        for row in rows
    ]


def summarize(items: Sequence[Prediction]) -> dict[str, float | int]:
    errors = [item.error_kg for item in items]
    return {
        "animals": len(items),
        "mae_kg": sum(abs(value) for value in errors) / len(items),
        "rmse_kg": math.sqrt(sum(value**2 for value in errors) / len(items)),
        "bias_kg": sum(errors) / len(items),
        "mape_pct": sum(100 * item.absolute_error_kg / item.actual_kg for item in items)
        / len(items),
        "overestimated_animals": sum(value > 0 for value in errors),
        "underestimated_animals": sum(value < 0 for value in errors),
    }


def summarize_by_weight_band(items: Sequence[Prediction]) -> list[dict[str, object]]:
    result = []
    for label, lower, upper in WEIGHT_BANDS:
        group = [item for item in items if lower <= item.actual_kg < upper]
        if group:
            result.append({"weight_band": label, **summarize(group)})
    return result


def _point(value: float, low: float, high: float, extent: int) -> float:
    return 60 + (value - low) / (high - low) * (extent - 100)


def _chart(items: Sequence[Prediction], *, residual: bool, label: str) -> str:
    width, height = 760, 520
    actual = [item.actual_kg for item in items]
    y_values = [item.error_kg if residual else item.predicted_kg for item in items]
    x_low = math.floor((min(actual) - 20) / 50) * 50
    x_high = math.ceil((max(actual) + 20) / 50) * 50
    if residual:
        magnitude = math.ceil(max(abs(min(y_values)), abs(max(y_values))) / 25) * 25
        y_low, y_high = -magnitude, magnitude
        title, y_label = (
            f"Resíduos por peso real — {label}",
            "Erro: previsto − real (kg)",
        )
    else:
        all_weights = actual + y_values
        y_low = x_low = math.floor((min(all_weights) - 20) / 50) * 50
        y_high = x_high = math.ceil((max(all_weights) + 20) / 50) * 50
        title, y_label = f"Peso real × peso previsto — {label}", "Peso previsto (kg)"
    parts = [
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#444;stroke-width:1.5}</style>",
        (
            '<text x="380" y="28" text-anchor="middle" font-size="20" '
            f'font-weight="bold">{title}</text>'
        ),
        '<line class="axis" x1="60" y1="460" x2="720" y2="460"/>',
        '<line class="axis" x1="60" y1="50" x2="60" y2="460"/>',
    ]
    if residual:
        zero = height - _point(0, y_low, y_high, height)
        parts.append(
            f'<line x1="60" y1="{zero:.1f}" x2="720" y2="{zero:.1f}" '
            'stroke="#777" stroke-dasharray="6 5"/>'
        )
    else:
        x1, x2 = (
            _point(x_low, x_low, x_high, width),
            _point(x_high, x_low, x_high, width),
        )
        y1, y2 = (
            height - _point(y_low, y_low, y_high, height),
            height - _point(y_high, y_low, y_high, height),
        )
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#777" stroke-dasharray="6 5"/>'
        )
    for item, y_value in zip(items, y_values, strict=True):
        x = _point(item.actual_kg, x_low, x_high, width)
        y = height - _point(y_value, y_low, y_high, height)
        color = "#c0392b" if item.absolute_error_kg >= 50 else "#1769aa"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}">'
            f"<title>{item.animal_id}: real {item.actual_kg:.0f}, "
            f"previsto {item.predicted_kg:.1f}, erro {item.error_kg:+.1f} kg"
            "</title></circle>"
        )
    parts.extend(
        [
            '<text x="390" y="510" text-anchor="middle" font-size="14">'
            "Peso real (kg)</text>",
            '<text x="16" y="255" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 16 255)">{y_label}</text>',
        ]
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )


def _report(
    label: str,
    overall: dict[str, object],
    sensitivity: dict[str, object],
    bands: Sequence[dict[str, object]],
    worst: Sequence[Prediction],
) -> str:
    lines = [
        f"# Análise de erros — {label}",
        "",
        "## Resumo",
        "",
        f"- {overall['animals']} animais no teste;",
        f"- MAE: {overall['mae_kg']:.2f} kg; RMSE: {overall['rmse_kg']:.2f} kg;",
        f"- MAPE: {overall['mape_pct']:.2f}%; viés: {overall['bias_kg']:+.2f} kg;",
        f"- {overall['overestimated_animals']} superestimados e "
        f"{overall['underestimated_animals']} subestimados.",
        "",
        "## Sensibilidade aos animais abaixo de 350 kg",
        "",
        "Os dois animais mais leves concentram os dois maiores erros. Ao relatar "
        "separadamente os 18 animais com pelo menos 350 kg:",
        "",
        f"- MAE: {sensitivity['mae_kg']:.2f} kg;",
        f"- MAPE: {sensitivity['mape_pct']:.2f}%;",
        f"- viés: {sensitivity['bias_kg']:+.2f} kg.",
        "",
        "## Erro por faixa de peso",
        "",
        "| Faixa | n | MAE (kg) | MAPE | Viés (kg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in bands:
        lines.append(
            f"| {row['weight_band']} | {row['animals']} | "
            f"{row['mae_kg']:.2f} | {row['mape_pct']:.2f}% | "
            f"{row['bias_kg']:+.2f} |"
        )
    lines += [
        "",
        "## Cinco maiores erros",
        "",
        "| Animal | Real | Previsto | Erro |",
        "|---|---:|---:|---:|",
    ]
    for item in worst:
        lines.append(
            f"| {item.animal_id} | {item.actual_kg:.0f} kg | "
            f"{item.predicted_kg:.1f} kg | {item.error_kg:+.1f} kg |"
        )
    lines += [
        "",
        "## Leitura",
        "",
        "O sinal do viés e sua concentração por faixa devem orientar a próxima "
        "iteração. Com apenas 20 animais no teste, o resultado ainda é preliminar.",
        "",
        "![Peso real versus previsto](scatter_actual_vs_predicted.svg)",
        "",
        "![Resíduos](residuals_by_actual_weight.svg)",
        "",
    ]
    return "\n".join(lines)


def write_analysis(
    items: Sequence[Prediction], output_dir: str | Path, *, label: str
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    overall = summarize(items)
    sensitivity = summarize([item for item in items if item.actual_kg >= 350])
    bands = summarize_by_weight_band(items)
    worst = sorted(items, key=lambda item: item.absolute_error_kg, reverse=True)[:5]
    payload = {
        "overall": overall,
        "sensitivity_actual_at_least_350kg": sensitivity,
        "by_weight_band": bands,
        "largest_errors": [
            asdict(item) | {"error_kg": item.error_kg} for item in worst
        ],
    }
    (destination / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (destination / "error_by_weight_band.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)
    (destination / "scatter_actual_vs_predicted.svg").write_text(
        _chart(items, residual=False, label=label), encoding="utf-8"
    )
    (destination / "residuals_by_actual_weight.svg").write_text(
        _chart(items, residual=True, label=label), encoding="utf-8"
    )
    (destination / "error_analysis.md").write_text(
        _report(label, overall, sensitivity, bands, worst), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa erros de regressão de peso")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    write_analysis(
        load_predictions(args.predictions), args.output_dir, label=args.label
    )
    print(f"Análise gravada em {args.output_dir}")


if __name__ == "__main__":
    main()
