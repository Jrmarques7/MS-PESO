from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, stdev

METRICS = ("mae_kg", "rmse_kg", "mape_pct", "bias_kg", "r2")


def load_model_metrics(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: float(payload["model"][name]) for name in METRICS}


def summarize(paths: Sequence[str | Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("São necessárias pelo menos duas execuções")
    runs = [load_model_metrics(path) for path in paths]
    metrics = {}
    for name in METRICS:
        values = [run[name] for run in runs]
        metrics[name] = {
            "values": values,
            "mean": fmean(values),
            "sample_std": stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {"runs": len(runs), "metrics": metrics}


def markdown_report(
    reference: dict[str, object],
    candidate: dict[str, object],
    *,
    reference_label: str,
    candidate_label: str,
) -> str:
    reference_metrics = reference["metrics"]
    candidate_metrics = candidate["metrics"]
    lines = [
        f"# Estabilidade entre seeds — {candidate_label} × {reference_label}",
        "",
        f"{reference['runs']} execuções por estratégia, usando a mesma divisão "
        "de animais.",
        "",
        "| Métrica | Referência média ± DP | Candidato média ± DP | "
        "Diferença das médias |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        reference_values = reference_metrics[name]
        candidate_values = candidate_metrics[name]
        difference = candidate_values["mean"] - reference_values["mean"]
        lines.append(
            f"| {name} | {reference_values['mean']:.2f} ± "
            f"{reference_values['sample_std']:.2f} | "
            f"{candidate_values['mean']:.2f} ± "
            f"{candidate_values['sample_std']:.2f} | {difference:+.2f} |"
        )
    lines.extend(
        [
            "",
            "O desvio-padrão descreve variação entre treinamentos, não incerteza "
            "populacional. Três seeds ainda formam uma amostra pequena.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume métricas entre seeds")
    parser.add_argument("--reference", required=True, nargs="+", type=Path)
    parser.add_argument("--candidate", required=True, nargs="+", type=Path)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    reference = summarize(args.reference)
    candidate = summarize(args.candidate)
    payload = {
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "reference": reference,
        "candidate": candidate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "seed_summary.md").write_text(
        markdown_report(
            reference,
            candidate,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
        ),
        encoding="utf-8",
    )
    print(f"Resumo gravado em {args.output_dir}")


if __name__ == "__main__":
    main()
