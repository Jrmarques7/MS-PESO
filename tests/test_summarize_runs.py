import json
import math

import pytest

from ms_peso.summarize_runs import summarize


def _write_metrics(path, mae: float) -> None:
    model = {
        "mae_kg": mae,
        "rmse_kg": mae + 1,
        "mape_pct": mae + 2,
        "bias_kg": mae + 3,
        "r2": mae + 4,
    }
    path.write_text(json.dumps({"model": model}), encoding="utf-8")


def test_summarizes_mean_and_sample_deviation(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_metrics(first, 10)
    _write_metrics(second, 20)

    result = summarize([first, second])

    assert result["metrics"]["mae_kg"]["mean"] == 15
    assert result["metrics"]["mae_kg"]["sample_std"] == pytest.approx(math.sqrt(50))


def test_requires_repeated_runs(tmp_path) -> None:
    with pytest.raises(ValueError, match="pelo menos duas"):
        summarize([tmp_path / "only.json"])
