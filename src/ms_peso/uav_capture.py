from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ms_peso.collection import AuthorizationRecord
from ms_peso.config import load_yaml_config


@dataclass(frozen=True)
class RangePolicy:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class UavCapturePolicy:
    policy_id: str
    policy_version: str
    required_flight_columns: tuple[str, ...]
    required_observation_columns: tuple[str, ...]
    allowed_views: tuple[str, ...]
    allowed_identity_statuses: tuple[str, ...]
    allowed_posture_statuses: tuple[str, ...]
    allowed_occlusion_statuses: tuple[str, ...]
    allowed_splits: tuple[str, ...]
    altitude_agl_m: RangePolicy
    ground_sample_distance_cm_px: RangePolicy
    weight_kg: RangePolicy
    require_scale_reference: bool
    require_commercial_training_rights: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class UavCaptureAuditReport:
    policy_id: str
    policy_version: str
    flights: int
    observations: int
    confirmed_animals: int
    regression_eligible_observations: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.valid else "rejected",
            "policy": {"id": self.policy_id, "version": self.policy_version},
            "summary": {
                "flights": self.flights,
                "observations": self.observations,
                "confirmed_animals": self.confirmed_animals,
                "regression_eligible_observations": (
                    self.regression_eligible_observations
                ),
            },
            "model_training_ready": False,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def _text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Política UAV sem texto {key!r}.")
    return value.strip()


def _text_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"Política UAV sem lista de textos {key!r}.")
    return tuple(item.strip() for item in value)


def _range(config: dict[str, Any], key: str) -> RangePolicy:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Política UAV sem faixa {key!r}.")
    minimum = value.get("min")
    maximum = value.get("max")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
    ):
        raise ValueError(f"Faixa UAV inválida para {key!r}.")
    result = RangePolicy(float(minimum), float(maximum))
    if (
        not math.isfinite(result.minimum)
        or not math.isfinite(result.maximum)
        or result.minimum <= 0
        or result.minimum >= result.maximum
    ):
        raise ValueError(f"Faixa UAV inválida para {key!r}.")
    return result


def load_uav_capture_policy(path: str | Path) -> UavCapturePolicy:
    config = load_yaml_config(path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão da política UAV não suportada.")
    for key in (
        "require_scale_reference",
        "require_commercial_training_rights",
    ):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"Política UAV sem booleano {key!r}.")
    return UavCapturePolicy(
        policy_id=_text(config, "policy_id"),
        policy_version=_text(config, "policy_version"),
        required_flight_columns=_text_list(config, "required_flight_columns"),
        required_observation_columns=_text_list(
            config, "required_observation_columns"
        ),
        allowed_views=_text_list(config, "allowed_views"),
        allowed_identity_statuses=_text_list(
            config, "allowed_identity_statuses"
        ),
        allowed_posture_statuses=_text_list(config, "allowed_posture_statuses"),
        allowed_occlusion_statuses=_text_list(
            config, "allowed_occlusion_statuses"
        ),
        allowed_splits=_text_list(config, "allowed_splits"),
        altitude_agl_m=_range(config, "altitude_agl_m"),
        ground_sample_distance_cm_px=_range(
            config, "ground_sample_distance_cm_px"
        ),
        weight_kg=_range(config, "weight_kg"),
        require_scale_reference=config["require_scale_reference"],
        require_commercial_training_rights=config[
            "require_commercial_training_rights"
        ],
        limitations=_text_list(config, "limitations"),
    )


def read_uav_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifesto UAV não encontrado: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifesto UAV sem cabeçalho: {csv_path}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"Manifesto UAV vazio: {csv_path}")
    return rows


def _timestamp(value: str, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} deve usar ISO 8601 com fuso horário") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} deve incluir o fuso horário")
    return result


def _number(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} deve ser numérico") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} deve ser finito")
    return result


def _bool(value: str, field: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} deve ser true ou false")


def _in_range(value: float, policy: RangePolicy) -> bool:
    return policy.minimum <= value <= policy.maximum


def _resolve(path: str, manifest_path: Path, data_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = data_root if data_root is not None else manifest_path.parent
    return (base / candidate).resolve()


def audit_uav_capture(
    flight_rows: list[dict[str, str]],
    observation_rows: list[dict[str, str]],
    authorizations: dict[str, AuthorizationRecord],
    policy: UavCapturePolicy,
    *,
    flights_manifest_path: Path,
    observations_manifest_path: Path,
    data_root: Path | None,
    check_files: bool,
) -> UavCaptureAuditReport:
    errors: list[str] = []
    warnings: list[str] = []
    flight_headers = set(flight_rows[0]) if flight_rows else set()
    observation_headers = set(observation_rows[0]) if observation_rows else set()
    missing_flight = set(policy.required_flight_columns) - flight_headers
    missing_observation = set(policy.required_observation_columns) - observation_headers
    if missing_flight:
        errors.append("Colunas de voo ausentes: " + ", ".join(sorted(missing_flight)))
    if missing_observation:
        errors.append(
            "Colunas de observação ausentes: "
            + ", ".join(sorted(missing_observation))
        )

    flights: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(flight_rows, start=2):
        prefix = f"voos linha {line_number}"
        flight_id = row.get("flight_id", "")
        if not flight_id:
            errors.append(f"{prefix}: flight_id vazio")
        elif flight_id in flights:
            errors.append(f"{prefix}: flight_id duplicado: {flight_id}")
        else:
            flights[flight_id] = row
        try:
            _timestamp(row.get("captured_at", ""), "captured_at")
            altitude = _number(row.get("altitude_agl_m", ""), "altitude_agl_m")
            gsd = _number(
                row.get("ground_sample_distance_cm_px", ""),
                "ground_sample_distance_cm_px",
            )
            if not _in_range(altitude, policy.altitude_agl_m):
                errors.append(f"{prefix}: altitude_agl_m fora da política")
            if not _in_range(gsd, policy.ground_sample_distance_cm_px):
                errors.append(
                    f"{prefix}: ground_sample_distance_cm_px fora da política"
                )
            if row.get("view", "") not in policy.allowed_views:
                errors.append(f"{prefix}: view deve ser top")
            if policy.require_scale_reference and not row.get("scale_reference", ""):
                errors.append(f"{prefix}: scale_reference é obrigatório")
            rights = _bool(
                row.get("commercial_training_allowed", ""),
                "commercial_training_allowed",
            )
            if policy.require_commercial_training_rights and not rights:
                errors.append(f"{prefix}: treinamento comercial não autorizado")
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

        authorization = authorizations.get(row.get("authorization_id", ""))
        if authorization is None:
            errors.append(f"{prefix}: authorization_id inexistente")
        elif authorization.farm_id != row.get("farm_id", ""):
            errors.append(f"{prefix}: autorização pertence a outra fazenda")
        elif authorization.status != "approved":
            errors.append(f"{prefix}: autorização não aprovada")
        elif policy.require_commercial_training_rights and (
            not authorization.allows_model_training
            or not authorization.allows_commercial_use
        ):
            errors.append(f"{prefix}: autorização não cobre o uso exigido")

        if check_files:
            source = _resolve(
                row.get("source_path", ""), flights_manifest_path, data_root
            )
            if not source.is_file():
                errors.append(f"{prefix}: source_path não encontrado")

    observation_ids: set[str] = set()
    animal_splits: dict[str, str] = {}
    confirmed_animals: set[str] = set()
    regression_eligible = 0
    for line_number, row in enumerate(observation_rows, start=2):
        prefix = f"observações linha {line_number}"
        observation_id = row.get("observation_id", "")
        if not observation_id:
            errors.append(f"{prefix}: observation_id vazio")
        elif observation_id in observation_ids:
            errors.append(f"{prefix}: observation_id duplicado: {observation_id}")
        observation_ids.add(observation_id)

        flight = flights.get(row.get("flight_id", ""))
        if flight is None:
            errors.append(f"{prefix}: flight_id inexistente")
        identity_status = row.get("identity_status", "")
        if identity_status not in policy.allowed_identity_statuses:
            errors.append(f"{prefix}: identity_status inválido")
        posture = row.get("posture_status", "")
        occlusion = row.get("occlusion_status", "")
        if posture not in policy.allowed_posture_statuses:
            errors.append(f"{prefix}: posture_status inválido")
        if occlusion not in policy.allowed_occlusion_statuses:
            errors.append(f"{prefix}: occlusion_status inválido")

        if check_files:
            segment = _resolve(
                row.get("segment_path", ""),
                observations_manifest_path,
                data_root,
            )
            if not segment.is_file():
                errors.append(f"{prefix}: segment_path não encontrado")

        if identity_status != "confirmed":
            if row.get("weight_kg", "") or row.get("split", ""):
                errors.append(
                    f"{prefix}: identidade desconhecida não pode ter peso ou split"
                )
            continue

        animal_id = row.get("animal_id", "")
        split = row.get("split", "")
        if not animal_id:
            errors.append(f"{prefix}: animal_id obrigatório para identidade confirmada")
        else:
            confirmed_animals.add(animal_id)
        if split not in policy.allowed_splits:
            errors.append(f"{prefix}: split inválido")
        elif animal_id:
            previous_split = animal_splits.setdefault(animal_id, split)
            if previous_split != split:
                errors.append(f"{prefix}: animal aparece em mais de um split")
        try:
            weight = _number(row.get("weight_kg", ""), "weight_kg")
            _timestamp(row.get("weighed_at", ""), "weighed_at")
            if not _in_range(weight, policy.weight_kg):
                errors.append(f"{prefix}: weight_kg fora da política")
            if flight is not None and row.get("weighing_event_id", "") != flight.get(
                "event_id", ""
            ):
                errors.append(f"{prefix}: pesagem não pertence ao evento do voo")
            if posture == "standing" and occlusion == "clear":
                regression_eligible += 1
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

    if not check_files:
        warnings.append("Arquivos brutos e segmentos não foram verificados.")
    warnings.append(
        "A auditoria valida o contrato, não suficiência estatística nem desempenho."
    )
    return UavCaptureAuditReport(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        flights=len(flights),
        observations=len(observation_rows),
        confirmed_animals=len(confirmed_animals),
        regression_eligible_observations=regression_eligible,
        errors=tuple(errors),
        warnings=tuple(warnings),
        limitations=policy.limitations,
    )
