from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ms_peso.init_collection_batch import initialize_collection_batch


def _template(path: Path, header: list[str], example: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(example)


def test_initialize_collection_batch_creates_only_empty_templates(
    tmp_path: Path,
) -> None:
    video_template = tmp_path / "video.csv"
    auth_template = tmp_path / "auth.csv"
    _template(video_template, ["video_path", "animal_id"], ["example.mp4", "a1"])
    _template(auth_template, ["authorization_id", "farm_id"], ["auth1", "farm1"])

    result = initialize_collection_batch(
        data_root=tmp_path / "data",
        batch_id="batch_20260823_001",
        farm_id="farm_001",
        video_template=video_template,
        authorization_template=auth_template,
    )

    batch = tmp_path / "data/interim/pasture/batch_20260823_001"
    assert (tmp_path / "data/raw/pasture/farm_001").is_dir()
    assert (batch / "pasture_video_manifest.csv").read_text().splitlines() == [
        "video_path,animal_id"
    ]
    assert (batch / "authorization_registry.csv").read_text().splitlines() == [
        "authorization_id,farm_id"
    ]
    metadata = json.loads((batch / "batch_metadata.json").read_text())
    assert metadata["status"] == "open_for_collection"
    assert result["batch_id"] == "batch_20260823_001"


def test_initialize_collection_batch_refuses_to_overwrite(tmp_path: Path) -> None:
    video_template = tmp_path / "video.csv"
    auth_template = tmp_path / "auth.csv"
    _template(video_template, ["video_path"], ["example.mp4"])
    _template(auth_template, ["authorization_id"], ["auth1"])
    kwargs = {
        "data_root": tmp_path / "data",
        "batch_id": "batch_001",
        "farm_id": "farm_001",
        "video_template": video_template,
        "authorization_template": auth_template,
    }
    initialize_collection_batch(**kwargs)

    with pytest.raises(FileExistsError, match="não será sobrescrito"):
        initialize_collection_batch(**kwargs)


@pytest.mark.parametrize("unsafe", ["../batch", "batch 1", "", "a/b"])
def test_initialize_collection_batch_rejects_unsafe_ids(
    tmp_path: Path, unsafe: str
) -> None:
    with pytest.raises(ValueError):
        initialize_collection_batch(
            data_root=tmp_path / "data",
            batch_id=unsafe,
            farm_id="farm_001",
            video_template=tmp_path / "unused.csv",
            authorization_template=tmp_path / "unused-auth.csv",
        )
