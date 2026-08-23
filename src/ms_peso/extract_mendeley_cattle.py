from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    size_bytes: int
    sha256: str
    default_destination: str


ARCHIVES = {
    "multiview": ArchiveSpec(
        filename="multiview_cow_weight_v1.zip",
        size_bytes=961_978_093,
        sha256="962399c12fbaa655abd314fa9037ada3854b8ce6ecc0af2847e9e5c5f27696ec",
        default_destination="data/raw/mendeley/multiview_v1",
    ),
    "horqin": ArchiveSpec(
        filename="horqin_side_back_v3.zip",
        size_bytes=2_517_767_623,
        sha256="47773895edc1123cb3057061521227d19450a36a396c04a12a10de24a72dfaa2",
        default_destination="data/raw/mendeley/horqin_v3",
    ),
}


def calculate_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: str | Path, spec: ArchiveSpec) -> None:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {archive_path}")
    if archive_path.stat().st_size != spec.size_bytes:
        raise ValueError(
            f"Tamanho inesperado para {archive_path.name}: "
            f"{archive_path.stat().st_size}; esperado {spec.size_bytes}."
        )
    actual_sha256 = calculate_sha256(archive_path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"SHA-256 inesperado para {archive_path.name}: {actual_sha256}."
        )


def safe_extract_zip(archive: str | Path, destination: str | Path) -> int:
    """Extrai um ZIP novo, rejeitando travessia de diretório e links simbólicos."""
    archive_path = Path(archive)
    destination_path = Path(destination)
    if destination_path.exists() and not destination_path.is_dir():
        raise FileExistsError(f"Destino existe e não é uma pasta: {destination_path}")
    if destination_path.exists() and any(destination_path.iterdir()):
        raise FileExistsError(
            f"Destino já contém arquivos e não será sobrescrito: {destination_path}"
        )
    destination_path.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination_path.resolve()
    extracted_files = 0

    with zipfile.ZipFile(archive_path) as compressed:
        for member in compressed.infolist():
            relative_path = _validated_member_path(member)
            target = (destination_resolved / Path(*relative_path.parts)).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise ValueError(f"Caminho inseguro no ZIP: {member.filename}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with compressed.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted_files += 1
    return extracted_files


def _validated_member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    normalized = member.filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    mode = member.external_attr >> 16
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or ":" in path.parts[0]
        or stat.S_ISLNK(mode)
    ):
        raise ValueError(f"Entrada insegura no ZIP: {member.filename}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica e extrai com segurança os datasets bovinos Mendeley."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ARCHIVES))
    parser.add_argument(
        "--archives-root", default="data/raw/mendeley/archives", type=Path
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ARCHIVES[args.dataset]
    archive = args.archives_root / spec.filename
    verify_archive(archive, spec)
    print(f"Arquivo verificado: {archive} ({spec.sha256})")
    if args.verify_only:
        return
    destination = args.destination or Path(spec.default_destination)
    extracted = safe_extract_zip(archive, destination)
    print(f"Extração concluída em {destination}; arquivos: {extracted}")


if __name__ == "__main__":
    main()
