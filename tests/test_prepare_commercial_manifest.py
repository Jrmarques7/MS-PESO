import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ms_peso.artifacts import save_json
from ms_peso.collection_snapshot import build_collection_snapshot
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import read_manifest, write_manifest
from ms_peso.prepare_commercial_manifest import main


def _prepare_sealed_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index in range(20):
        image_path = tmp_path / f"cow_{index:03d}.png"
        Image.new(
            "RGB",
            (64, 64),
            (index * 10, 255 - index * 10, 50 + index),
        ).save(image_path)
        rows.append(
            {
                "image_path": image_path.name,
                "animal_id": f"nelore_{index:03d}",
                "event_id": f"event_{index:03d}",
                "weight_kg": str(200 + index * 20),
                "view": "left",
                "farm_id": "farm_001",
            }
        )
    snapshot = build_collection_snapshot(
        rows,
        manifest_path=tmp_path / "source.csv",
        image_root=tmp_path,
        near_duplicate_hamming_distance=0,
    )
    sealed_manifest = tmp_path / "sealed.csv"
    write_manifest(snapshot.rows, sealed_manifest)
    seal_report = tmp_path / "seal_report.json"
    save_json(
        seal_report,
        {
            "status": "passed",
            "stage": "sealed",
            "snapshot_id": snapshot.snapshot_id,
            "provenance": {
                "sealed_manifest_sha256": calculate_sha256(sealed_manifest)
            },
        },
    )
    return sealed_manifest, seal_report


def _arguments(
    tmp_path: Path, sealed_manifest: Path, seal_report: Path
) -> list[str]:
    return [
        "ms_peso.prepare_commercial_manifest",
        "--input",
        str(sealed_manifest),
        "--snapshot-report",
        str(seal_report),
        "--image-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "commercial_split.csv"),
        "--output-report",
        str(tmp_path / "split_report.json"),
    ]


def test_cli_creates_four_independent_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed_manifest, seal_report = _prepare_sealed_snapshot(tmp_path)
    monkeypatch.setattr(
        sys, "argv", _arguments(tmp_path, sealed_manifest, seal_report)
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    rows = read_manifest(tmp_path / "commercial_split.csv")
    assert payload["status"] == "passed"
    assert set(payload["splits"]) == {"train", "val", "calibration", "test"}
    assert payload["splits"]["calibration"]["animals"] == 2
    assert {row["split"] for row in rows} == {
        "train",
        "val",
        "calibration",
        "test",
    }


def test_cli_rejects_image_changed_after_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed_manifest, seal_report = _prepare_sealed_snapshot(tmp_path)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(tmp_path / "cow_000.png")
    monkeypatch.setattr(
        sys, "argv", _arguments(tmp_path, sealed_manifest, seal_report)
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    payload = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 2
    assert payload["status"] == "rejected"
    assert "conteúdo da imagem foi alterado" in payload["errors"][0]
    assert not (tmp_path / "commercial_split.csv").exists()
