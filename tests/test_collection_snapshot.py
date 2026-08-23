import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ms_peso.collection_snapshot import build_collection_snapshot
from ms_peso.manifest import write_manifest
from ms_peso.seal_collection import main


def snapshot_row(image_path: str, index: int) -> dict[str, str]:
    return {
        "image_path": image_path,
        "farm_id": "farm_001",
        "animal_id": f"nelore_{index:03d}",
        "event_id": f"event_{index:03d}",
        "view": "left",
    }


def test_blocks_files_with_identical_content(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), (100, 120, 140))
    image.save(tmp_path / "first.png")
    image.save(tmp_path / "second.png")

    snapshot = build_collection_snapshot(
        [snapshot_row("first.png", 1), snapshot_row("second.png", 2)],
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        near_duplicate_hamming_distance=4,
    )

    assert snapshot.valid is False
    assert len(snapshot.exact_duplicates) == 1
    assert snapshot.to_report()["status"] == "rejected"


def test_flags_visually_similar_files_without_blocking(tmp_path: Path) -> None:
    first = Image.new("L", (128, 64), 128)
    second = first.copy()
    second.putpixel((0, 0), 129)
    first.save(tmp_path / "first.png")
    second.save(tmp_path / "second.png")

    snapshot = build_collection_snapshot(
        [snapshot_row("first.png", 1), snapshot_row("second.png", 2)],
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        near_duplicate_hamming_distance=4,
    )

    assert snapshot.valid is True
    assert snapshot.exact_duplicates == ()
    assert len(snapshot.near_duplicates) == 1
    assert snapshot.near_duplicates[0].hamming_distance <= 4


def test_snapshot_id_is_independent_of_source_row_order(tmp_path: Path) -> None:
    Image.new("RGB", (64, 64), (30, 40, 50)).save(tmp_path / "first.png")
    Image.new("RGB", (64, 64), (200, 180, 160)).save(tmp_path / "second.png")
    rows = [snapshot_row("first.png", 1), snapshot_row("second.png", 2)]

    first = build_collection_snapshot(
        rows,
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        near_duplicate_hamming_distance=4,
    )
    second = build_collection_snapshot(
        list(reversed(rows)),
        manifest_path=tmp_path / "manifest.csv",
        image_root=tmp_path,
        near_duplicate_hamming_distance=4,
    )

    assert first.snapshot_id == second.snapshot_id
    assert all("image_sha256" in row for row in first.rows)
    assert all("image_dhash" in row for row in first.rows)


def _write_checkerboard(path: Path) -> None:
    image = Image.new("RGB", (800, 500), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    for y in range(0, 500, 16):
        for x in range(0, 800, 16):
            if (x // 16 + y // 16) % 2 == 0:
                draw.rectangle((x, y, x + 15, y + 15), fill=(224, 224, 224))
    image.save(path)


def collection_row() -> dict[str, str]:
    return {
        "image_path": "cow.jpg",
        "animal_id": "nelore_001",
        "event_id": "event_001",
        "weight_kg": "425.5",
        "view": "left",
        "breed": "nelore",
        "sex": "female",
        "farm_id": "farm_001",
        "lot_id": "lot_001",
        "captured_at": "2026-08-22T09:02:00-03:00",
        "weighed_at": "2026-08-22T09:00:00-03:00",
        "camera_id": "camera_001",
        "scale_id": "scale_001",
        "quality": "accepted",
        "scale_marker": "true",
        "authorization_id": "auth_001",
        "commercial_training_allowed": "true",
        "notes": "",
    }


def test_cli_seals_collection_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_checkerboard(tmp_path / "cow.jpg")
    manifest = tmp_path / "manifest.csv"
    write_manifest([collection_row()], manifest)
    registry = tmp_path / "authorizations.csv"
    registry.write_text(
        """authorization_id,farm_id,status,effective_from,effective_until,allows_model_training,allows_commercial_use,allows_data_sharing,document_reference,notes
auth_001,farm_001,approved,2026-08-01,,true,true,false,secure://auth_001,
""",
        encoding="utf-8",
    )
    sealed = tmp_path / "sealed.csv"
    report = tmp_path / "snapshot.json"
    arguments = [
        "ms_peso.seal_collection",
        "--manifest",
        str(manifest),
        "--authorizations",
        str(registry),
        "--image-root",
        str(tmp_path),
        "--output-manifest",
        str(sealed),
        "--output-report",
        str(report),
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    main()

    payload = json.loads(capsys.readouterr().out)
    sealed_content = sealed.read_text(encoding="utf-8")
    report_content = report.read_text(encoding="utf-8")
    assert payload["status"] == "passed"
    assert payload["stage"] == "sealed"
    assert payload["snapshot_id"]
    assert "image_sha256" in sealed_content
    assert payload["provenance"]["sealed_manifest_sha256"]

    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as exit_info:
        main()

    rejection = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 2
    assert rejection["status"] == "rejected"
    assert "já existe" in rejection["errors"][0]
    assert sealed.read_text(encoding="utf-8") == sealed_content
    assert report.read_text(encoding="utf-8") == report_content
