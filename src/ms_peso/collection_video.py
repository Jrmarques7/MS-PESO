from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ms_peso.collection import CollectionPolicy
from ms_peso.image_quality import ImageQualityPolicy, assess_image_quality
from ms_peso.service.video_frames import VideoValidationError, extract_uniform_frames
from ms_peso.service.video_policy import VideoInferencePolicy
from ms_peso.service.video_selection import (
    AssessedFrame,
    FrameSelectionError,
    select_technical_frames,
    technical_quality_score,
)

PASTURE_VIDEO_REQUIRED_COLUMNS = (
    "video_path",
    "animal_id",
    "event_id",
    "weight_kg",
    "view",
    "breed",
    "sex",
    "farm_id",
    "lot_id",
    "captured_at",
    "weighed_at",
    "camera_id",
    "scale_id",
    "primary_full_body",
    "primary_lateral",
    "quality",
    "scale_marker",
    "authorization_id",
    "commercial_training_allowed",
    "notes",
)

OUTPUT_COLUMNS = (
    "image_path",
    "animal_id",
    "event_id",
    "weight_kg",
    "view",
    "breed",
    "sex",
    "farm_id",
    "lot_id",
    "captured_at",
    "weighed_at",
    "camera_id",
    "scale_id",
    "quality",
    "scale_marker",
    "authorization_id",
    "commercial_training_allowed",
    "notes",
    "source_video_path",
    "source_frame_index",
    "source_timestamp_seconds",
    "technical_quality_score",
    "source_video_duration_seconds",
    "technical_quality_policy_id",
    "technical_quality_policy_version",
    "video_selection_policy_id",
    "video_selection_policy_version",
)

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class VideoFrameCandidate:
    line_number: int
    source_row: dict[str, str]
    temporary_image: Path
    frame_index: int
    timestamp_seconds: float
    technical_score: float
    duration_seconds: float


def read_pasture_video_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto de vídeos não encontrado: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("O manifesto de vídeos não possui cabeçalho CSV.")
        missing = set(PASTURE_VIDEO_REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "Colunas de vídeo ausentes: " + ", ".join(sorted(missing))
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("O manifesto de vídeos está vazio.")
    return rows


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} deve ser true ou false")


def _parse_timestamp(value: str, *, field: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} deve usar ISO 8601 com fuso horário") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} deve incluir o fuso horário")
    return result


def _validate_safe_id(value: str, *, field: str) -> None:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} possui identificador vazio ou inseguro")


def _resolve_video_path(value: str, video_root: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("video_path deve ser relativo ao --video-root")
    root = video_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("video_path tenta sair do --video-root")
    if not candidate.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {candidate}")
    return candidate


def _validate_source_row(row: dict[str, str], policy: CollectionPolicy) -> None:
    for field in (
        "animal_id",
        "event_id",
        "farm_id",
        "lot_id",
        "camera_id",
        "scale_id",
        "authorization_id",
    ):
        _validate_safe_id(row[field], field=field)

    try:
        weight = float(row["weight_kg"])
    except ValueError as exc:
        raise ValueError("weight_kg deve vir da balança e ser numérico") from exc
    if not math.isfinite(weight) or not (
        policy.min_weight_kg <= weight <= policy.max_weight_kg
    ):
        raise ValueError(
            f"weight_kg deve estar entre {policy.min_weight_kg:g} e "
            f"{policy.max_weight_kg:g} kg"
        )
    if row["view"] not in policy.allowed_views:
        raise ValueError(f"view não permitido: {row['view']!r}")
    if row["breed"] not in policy.allowed_breeds:
        raise ValueError(f"breed não permitido: {row['breed']!r}")
    if row["sex"] not in policy.allowed_sexes:
        raise ValueError(f"sex não permitido: {row['sex']!r}")
    if row["quality"] != policy.required_quality:
        raise ValueError(
            f"quality do vídeo deve ser {policy.required_quality!r} após revisão humana"
        )
    if not _parse_bool(row["primary_full_body"], field="primary_full_body"):
        raise ValueError("o animal principal não está com o corpo inteiro")
    if not _parse_bool(row["primary_lateral"], field="primary_lateral"):
        raise ValueError("o animal principal não está em pose lateral")
    scale_marker = _parse_bool(row["scale_marker"], field="scale_marker")
    if policy.require_scale_marker and not scale_marker:
        raise ValueError("o marcador de escala é obrigatório")
    commercial_training_allowed = _parse_bool(
        row["commercial_training_allowed"],
        field="commercial_training_allowed",
    )
    if policy.require_commercial_training_rights and not commercial_training_allowed:
        raise ValueError("o registro não autoriza treinamento comercial")

    captured_at = _parse_timestamp(row["captured_at"], field="captured_at")
    weighed_at = _parse_timestamp(row["weighed_at"], field="weighed_at")
    delta_minutes = abs((captured_at - weighed_at).total_seconds()) / 60
    if delta_minutes > policy.max_capture_weight_delta_minutes:
        raise ValueError(
            f"vídeo e pesagem estão separados por {delta_minutes:.1f} minutos"
        )


def _event_conflicts(rows: list[dict[str, str]]) -> set[int]:
    owners: dict[str, set[tuple[str, str]]] = defaultdict(set)
    memberships: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        event_id = row.get("event_id", "")
        owners[event_id].add((row.get("animal_id", ""), row.get("farm_id", "")))
        memberships[event_id].append(index)
    return {
        index
        for event_id, event_owners in owners.items()
        if len(event_owners) > 1
        for index in memberships[event_id]
    }


def _group_is_consistent(candidates: list[VideoFrameCandidate]) -> bool:
    fields = (
        "animal_id",
        "event_id",
        "weight_kg",
        "view",
        "breed",
        "sex",
        "farm_id",
        "lot_id",
        "weighed_at",
        "scale_id",
        "scale_marker",
        "authorization_id",
        "commercial_training_allowed",
    )
    signatures = {
        tuple(item.source_row[field] for field in fields) for item in candidates
    }
    return len(signatures) == 1


def _output_row(
    candidate: VideoFrameCandidate,
    *,
    image_path: str,
    quality_policy: ImageQualityPolicy,
    video_policy: VideoInferencePolicy,
) -> dict[str, str]:
    source = candidate.source_row
    source_notes = source.get("notes", "").strip()
    safety_note = "AUTO_FRAME_SELECTED; HUMAN_REVIEW_REQUIRED"
    notes = f"{source_notes}; {safety_note}" if source_notes else safety_note
    return {
        "image_path": image_path.replace("\\", "/"),
        "animal_id": source["animal_id"],
        "event_id": source["event_id"],
        "weight_kg": source["weight_kg"],
        "view": source["view"],
        "breed": source["breed"],
        "sex": source["sex"],
        "farm_id": source["farm_id"],
        "lot_id": source["lot_id"],
        "captured_at": source["captured_at"],
        "weighed_at": source["weighed_at"],
        "camera_id": source["camera_id"],
        "scale_id": source["scale_id"],
        "quality": "review",
        "scale_marker": source["scale_marker"],
        "authorization_id": source["authorization_id"],
        "commercial_training_allowed": source["commercial_training_allowed"],
        "notes": notes,
        "source_video_path": source["video_path"],
        "source_frame_index": str(candidate.frame_index),
        "source_timestamp_seconds": f"{candidate.timestamp_seconds:.6f}",
        "technical_quality_score": f"{candidate.technical_score:.6f}",
        "source_video_duration_seconds": f"{candidate.duration_seconds:.6f}",
        "technical_quality_policy_id": quality_policy.policy_id,
        "technical_quality_policy_version": quality_policy.policy_version,
        "video_selection_policy_id": video_policy.policy_id,
        "video_selection_policy_version": video_policy.policy_version,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def select_collection_frames(
    *,
    manifest_path: Path,
    video_root: Path,
    image_root: Path,
    output_directory: Path,
    collection_policy: CollectionPolicy,
    quality_policy: ImageQualityPolicy,
    video_policy: VideoInferencePolicy,
) -> dict[str, object]:
    if output_directory.exists():
        raise FileExistsError(
            f"A saída já existe e não será sobrescrita: {output_directory}"
        )
    image_root = image_root.resolve()
    final_output = output_directory.resolve()
    if not final_output.is_relative_to(image_root):
        raise ValueError("--output-directory deve ficar dentro de --image-root")
    rows = read_pasture_video_manifest(manifest_path)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}-preparing-",
            dir=output_directory.parent,
        )
    )
    candidate_directory = staging / "candidates"
    selected_directory = staging / "selected_frames"
    candidate_directory.mkdir()
    selected_directory.mkdir()
    rejections: list[dict[str, object]] = []
    candidates_by_group: dict[tuple[str, str], list[VideoFrameCandidate]] = (
        defaultdict(list)
    )
    conflicting_indices = _event_conflicts(rows)

    try:
        for index, row in enumerate(rows):
            line_number = index + 2
            if index in conflicting_indices:
                rejections.append(
                    {
                        "line": line_number,
                        "video_path": row.get("video_path", ""),
                        "code": "event_owner_conflict",
                        "detail": (
                            "event_id foi reutilizado por outro animal ou fazenda"
                        ),
                    }
                )
                continue
            try:
                _validate_source_row(row, collection_policy)
                video_path = _resolve_video_path(row["video_path"], video_root)
                extracted = extract_uniform_frames(
                    video_path,
                    max_duration_seconds=video_policy.max_duration_seconds,
                    max_frame_pixels=video_policy.max_frame_pixels,
                    sample_count=video_policy.sample_count,
                    minimum_decoded_frames=video_policy.min_valid_frames,
                )
                try:
                    assessed = []
                    for frame in extracted.frames:
                        quality = assess_image_quality(frame.path, quality_policy)
                        assessed.append(
                            AssessedFrame(
                                frame=frame,
                                quality=quality,
                                technical_score=technical_quality_score(quality),
                            )
                        )
                    selection = select_technical_frames(tuple(assessed), video_policy)
                    best = max(
                        selection.selected,
                        key=lambda item: (
                            item.technical_score,
                            -item.frame.timestamp_seconds,
                        ),
                    )
                    temporary_image = candidate_directory / f"line-{line_number}.png"
                    shutil.copyfile(best.frame.path, temporary_image)
                    candidates_by_group[(row["event_id"], row["view"])].append(
                        VideoFrameCandidate(
                            line_number=line_number,
                            source_row=row,
                            temporary_image=temporary_image,
                            frame_index=best.frame.frame_index,
                            timestamp_seconds=best.frame.timestamp_seconds,
                            technical_score=best.technical_score,
                            duration_seconds=extracted.metadata.duration_seconds,
                        )
                    )
                finally:
                    extracted.remove()
            except (
                FileNotFoundError,
                FrameSelectionError,
                ValueError,
                VideoValidationError,
            ) as exc:
                rejections.append(
                    {
                        "line": line_number,
                        "video_path": row.get("video_path", ""),
                        "code": getattr(exc, "code", "invalid_source_row"),
                        "detail": str(exc),
                    }
                )

        output_rows: list[dict[str, str]] = []
        for (event_id, view), group in sorted(candidates_by_group.items()):
            if not _group_is_consistent(group):
                for candidate in group:
                    rejections.append(
                        {
                            "line": candidate.line_number,
                            "video_path": candidate.source_row["video_path"],
                            "code": "event_metadata_conflict",
                            "detail": (
                                "vídeos do mesmo evento/vista possuem metadados "
                                "de identidade ou pesagem divergentes"
                            ),
                        }
                    )
                continue
            chosen = max(
                group,
                key=lambda item: (item.technical_score, -item.line_number),
            )
            identity = "|".join(
                (
                    chosen.source_row["farm_id"],
                    chosen.source_row["animal_id"],
                    event_id,
                    view,
                )
            )
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            filename = f"{event_id[:48]}__{view}__{digest}.png"
            selected_path = selected_directory / filename
            shutil.copyfile(chosen.temporary_image, selected_path)
            final_image_path = final_output / "selected_frames" / filename
            relative_image_path = final_image_path.relative_to(image_root)
            output_rows.append(
                _output_row(
                    chosen,
                    image_path=str(relative_image_path),
                    quality_policy=quality_policy,
                    video_policy=video_policy,
                )
            )

        shutil.rmtree(candidate_directory)
        status = (
            "rejected"
            if not output_rows
            else "completed_with_rejections"
            if rejections
            else "completed"
        )
        report: dict[str, object] = {
            "status": status,
            "summary": {
                "source_rows": len(rows),
                "candidate_videos": sum(
                    len(group) for group in candidates_by_group.values()
                ),
                "selected_images": len(output_rows),
                "rejected_rows": len(rejections),
            },
            "safety": {
                "weight_source": "scale_manifest_only",
                "weight_prediction_used": False,
                "output_quality": "review",
                "human_review_required": True,
                "one_image_per_event_view": True,
            },
            "policies": {
                "collection": {
                    "id": collection_policy.policy_id,
                    "version": collection_policy.policy_version,
                },
                "technical_quality": {
                    "id": quality_policy.policy_id,
                    "version": quality_policy.policy_version,
                },
                "video_selection": {
                    "id": video_policy.policy_id,
                    "version": video_policy.policy_version,
                    "status": video_policy.status,
                },
            },
            "rejections": rejections,
        }
        if output_rows:
            _write_csv(staging / "pilot_manifest.csv", output_rows)
        with (staging / "selection_report.json").open(
            "x", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        staging.rename(output_directory)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
