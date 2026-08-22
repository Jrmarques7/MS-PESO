from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SESSION_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")


@dataclass(frozen=True)
class NelloreUavInventory:
    """Resumo estrutural da versão longitudinal do dataset UAV de Nelore."""

    dataset_root: str
    sessions: dict[str, int]
    raw_images: int
    detection_images: int
    detection_labelme_json: int
    detection_yolo_labels: int
    feed_bunk_images: int
    feed_bunk_labelme_json: int
    detection_classes: dict[str, int]
    weight_metadata_files: tuple[str, ...]
    regression_ready: bool
    regression_blocker: str

    def to_dict(self) -> dict[str, object]:
        """Converte exclusivamente o inventário em dados serializáveis."""
        return asdict(self)


def resolve_nellore_uav_root(dataset_root: str | Path) -> Path:
    """Resolve exclusivamente o diretório raiz após a extração do ZIP."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {root}")

    wrapped_root = root / "NelloreBeefCattleDataset"
    if wrapped_root.is_dir():
        return wrapped_root
    return root


def scan_raw_uav_images(dataset_root: str | Path) -> dict[str, list[Path]]:
    """Localiza exclusivamente imagens UAV brutas agrupadas por sessão."""
    root = resolve_nellore_uav_root(dataset_root)
    sessions: dict[str, list[Path]] = {}
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or not SESSION_PATTERN.fullmatch(candidate.name):
            continue
        try:
            session = datetime.strptime(candidate.name, "%d-%m-%Y").date().isoformat()
        except ValueError as exc:
            raise ValueError(f"Data de sessão inválida: {candidate.name}") from exc

        altitude_directory = candidate / "15m"
        if not altitude_directory.is_dir():
            raise ValueError(f"Diretório 15m ausente na sessão {candidate.name}.")
        images = _image_files(altitude_directory)
        if not images:
            raise ValueError(f"Nenhuma imagem UAV na sessão {candidate.name}.")
        sessions[session] = images

    if not sessions:
        raise ValueError("Nenhuma sessão UAV no formato DD-MM-AAAA/15m foi encontrada.")
    return sessions


def scan_detection_annotations(dataset_root: str | Path) -> dict[str, object]:
    """Valida exclusivamente o corpus pareado de detecção de bovinos."""
    root = resolve_nellore_uav_root(dataset_root)
    cattle_root = root / "annotations_v2" / "gado"
    images_root = cattle_root / "images"
    labels_root = cattle_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            "Corpus annotations_v2/gado com images e labels não encontrado."
        )

    images = _image_files(images_root)
    labelme_files = sorted(images_root.rglob("*.json"))
    yolo_files = sorted(labels_root.rglob("*.txt"))
    image_keys = {_relative_stem(path, images_root) for path in images}
    labelme_keys = {_relative_stem(path, images_root) for path in labelme_files}
    yolo_keys = {_relative_stem(path, labels_root) for path in yolo_files}
    _require_same_keys("imagem", image_keys, "JSON LabelMe", labelme_keys)
    _require_same_keys("imagem", image_keys, "rótulo YOLO", yolo_keys)

    class_counts: Counter[str] = Counter()
    for label_path in yolo_files:
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "0":
                class_counts["cattle-back"] += 1
            elif fields[0] == "1":
                class_counts["cattle-head"] += 1
            else:
                class_counts[f"unknown:{fields[0]}"] += 1

    return {
        "images": len(images),
        "labelme_json": len(labelme_files),
        "yolo_labels": len(yolo_files),
        "classes": dict(sorted(class_counts.items())),
    }


def scan_feed_bunk_annotations(dataset_root: str | Path) -> dict[str, int]:
    """Valida exclusivamente os pares imagem/LabelMe da classe cocho."""
    root = resolve_nellore_uav_root(dataset_root)
    feed_bunk_root = root / "annotations_v2" / "cocho"
    if not feed_bunk_root.is_dir():
        raise FileNotFoundError("Corpus annotations_v2/cocho não encontrado.")

    images = _image_files(feed_bunk_root)
    labelme_files = sorted(feed_bunk_root.rglob("*.json"))
    image_keys = {_relative_stem(path, feed_bunk_root) for path in images}
    labelme_keys = {_relative_stem(path, feed_bunk_root) for path in labelme_files}
    _require_same_keys("imagem do cocho", image_keys, "JSON LabelMe", labelme_keys)
    return {"images": len(images), "labelme_json": len(labelme_files)}


def find_weight_metadata_files(dataset_root: str | Path) -> tuple[str, ...]:
    """Localiza exclusivamente arquivos candidatos a metadados de peso."""
    root = resolve_nellore_uav_root(dataset_root)
    metadata_suffixes = {".csv", ".tsv", ".xls", ".xlsx", ".parquet"}
    weight_terms = ("peso", "weight", "pesagem", "bodyweight")
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered_name = path.name.lower()
        if path.suffix.lower() in metadata_suffixes or any(
            term in lowered_name for term in weight_terms
        ):
            candidates.append(path.relative_to(root).as_posix())
    return tuple(sorted(candidates))


def inspect_nellore_uav_dataset(dataset_root: str | Path) -> NelloreUavInventory:
    """Compõe as inspeções independentes em um inventário do dataset."""
    root = resolve_nellore_uav_root(dataset_root)
    sessions = scan_raw_uav_images(root)
    detection = scan_detection_annotations(root)
    feed_bunk = scan_feed_bunk_annotations(root)
    weight_files = find_weight_metadata_files(root)
    blocker = (
        "O arquivo público não fornece animal_id persistente nem correspondência "
        "imagem-pesagem; não pode supervisionar regressão de peso."
    )
    return NelloreUavInventory(
        dataset_root=str(root.resolve()),
        sessions={key: len(value) for key, value in sorted(sessions.items())},
        raw_images=sum(len(value) for value in sessions.values()),
        detection_images=int(detection["images"]),
        detection_labelme_json=int(detection["labelme_json"]),
        detection_yolo_labels=int(detection["yolo_labels"]),
        feed_bunk_images=feed_bunk["images"],
        feed_bunk_labelme_json=feed_bunk["labelme_json"],
        detection_classes=dict(detection["classes"]),
        weight_metadata_files=weight_files,
        regression_ready=False,
        regression_blocker=blocker,
    )


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _relative_stem(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().lower()


def _require_same_keys(
    left_name: str,
    left: set[str],
    right_name: str,
    right: set[str],
) -> None:
    missing_right = sorted(left - right)
    missing_left = sorted(right - left)
    if missing_right or missing_left:
        details = []
        if missing_right:
            details.append(f"{right_name} ausente para: {missing_right[:5]}")
        if missing_left:
            details.append(f"{left_name} ausente para: {missing_left[:5]}")
        raise ValueError("; ".join(details))
