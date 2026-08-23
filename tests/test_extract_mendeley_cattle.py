import hashlib
import zipfile
from pathlib import Path

import pytest

from ms_peso.extract_mendeley_cattle import (
    ArchiveSpec,
    safe_extract_zip,
    verify_archive,
)


def test_verifies_and_extracts_archive(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as compressed:
        compressed.writestr("dataset/images/1.jpg", b"image")
        compressed.writestr("dataset/labels.csv", b"id,weight\n1,400\n")
    spec = ArchiveSpec(
        filename=archive.name,
        size_bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        default_destination="unused",
    )
    verify_archive(archive, spec)
    destination = tmp_path / "raw"
    assert safe_extract_zip(archive, destination) == 2
    assert (destination / "dataset" / "images" / "1.jpg").read_bytes() == b"image"


def test_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as compressed:
        compressed.writestr("../escape.txt", b"no")
    with pytest.raises(ValueError, match="insegura"):
        safe_extract_zip(archive, tmp_path / "raw")
    assert not (tmp_path / "escape.txt").exists()


def test_refuses_to_overwrite_nonempty_destination(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as compressed:
        compressed.writestr("file.txt", b"new")
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="não será sobrescrito"):
        safe_extract_zip(archive, destination)
    assert (destination / "existing.txt").read_text(encoding="utf-8") == "keep"
