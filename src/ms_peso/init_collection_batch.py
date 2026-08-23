from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _validate_id(value: str, *, field: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field} deve usar apenas letras, números, '_' ou '-', sem espaços."
        )
    return value


def _read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Template não encontrado: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), None)
    if not header or any(not item.strip() for item in header):
        raise ValueError(f"Template sem cabeçalho válido: {path}")
    return [item.strip() for item in header]


def _write_empty_csv(path: Path, header: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        csv.writer(file, lineterminator="\n").writerow(header)


def initialize_collection_batch(
    *,
    data_root: Path,
    batch_id: str,
    farm_id: str,
    video_template: Path,
    authorization_template: Path,
) -> dict[str, object]:
    batch_id = _validate_id(batch_id, field="batch_id")
    farm_id = _validate_id(farm_id, field="farm_id")
    data_root = data_root.resolve()
    batch_directory = data_root / "interim" / "pasture" / batch_id
    farm_directory = data_root / "raw" / "pasture" / farm_id
    if batch_directory.exists():
        raise FileExistsError(
            f"O lote já existe e não será sobrescrito: {batch_directory}"
        )

    video_header = _read_header(video_template)
    authorization_header = _read_header(authorization_template)
    batch_directory.parent.mkdir(parents=True, exist_ok=True)
    farm_directory.mkdir(parents=True, exist_ok=True)
    batch_directory.mkdir()
    try:
        video_manifest = batch_directory / "pasture_video_manifest.csv"
        authorization_registry = batch_directory / "authorization_registry.csv"
        metadata_path = batch_directory / "batch_metadata.json"
        _write_empty_csv(video_manifest, video_header)
        _write_empty_csv(authorization_registry, authorization_header)
        metadata = {
            "schema_version": 1,
            "batch_id": batch_id,
            "farm_id": farm_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "open_for_collection",
            "raw_video_root": str((data_root / "raw" / "pasture").resolve()),
            "farm_directory": str(farm_directory),
            "video_manifest": str(video_manifest),
            "authorization_registry": str(authorization_registry),
            "instructions": (
                "Preserve os originais, registre somente pesos de balança e "
                "não marque quality=accepted sem revisão humana."
            ),
        }
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except BaseException:
        shutil.rmtree(batch_directory, ignore_errors=True)
        raise
    return metadata


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um lote vazio e seguro para coleta lateral no pasto."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--farm-id", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--video-template",
        type=Path,
        default=Path("data/templates/pasture_video_manifest.csv"),
    )
    parser.add_argument(
        "--authorization-template",
        type=Path,
        default=Path("data/templates/authorization_registry.csv"),
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    try:
        result = initialize_collection_batch(
            data_root=args.data_root,
            batch_id=args.batch_id,
            farm_id=args.farm_id,
            video_template=args.video_template,
            authorization_template=args.authorization_template,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps({"status": "created", **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
