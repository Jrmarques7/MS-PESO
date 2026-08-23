from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ms_peso.config import load_yaml_config
from ms_peso.image_quality import ImageQualityPolicy, assess_image_quality
from ms_peso.manifest import resolve_image_path, validate_rows

AUTHORIZATION_COLUMNS = (
    "authorization_id",
    "farm_id",
    "status",
    "effective_from",
    "effective_until",
    "allows_model_training",
    "allows_commercial_use",
    "allows_data_sharing",
    "document_reference",
    "notes",
)


@dataclass(frozen=True)
class CollectionPolicy:
    policy_id: str
    policy_version: str
    required_columns: tuple[str, ...]
    allowed_views: tuple[str, ...]
    allowed_breeds: tuple[str, ...]
    allowed_sexes: tuple[str, ...]
    min_weight_kg: float
    max_weight_kg: float
    max_capture_weight_delta_minutes: int
    required_quality: str
    require_scale_marker: bool
    require_commercial_training_rights: bool


@dataclass(frozen=True)
class AuthorizationRecord:
    authorization_id: str
    farm_id: str
    status: str
    effective_from: date
    effective_until: date | None
    allows_model_training: bool
    allows_commercial_use: bool
    allows_data_sharing: bool
    document_reference: str


@dataclass(frozen=True)
class CollectionAuditReport:
    policy_id: str
    policy_version: str
    total_images: int
    animals: int
    events: int
    farms: int
    technical_quality_passed: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.valid else "rejected",
            "policy": {"id": self.policy_id, "version": self.policy_version},
            "summary": {
                "total_images": self.total_images,
                "animals": self.animals,
                "events": self.events,
                "farms": self.farms,
                "technical_quality_passed": self.technical_quality_passed,
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _required_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Política de coleta sem texto {key!r}.")
    return value.strip()


def _required_text_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"Política de coleta sem lista de textos {key!r}.")
    return tuple(item.strip() for item in value)


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Política de coleta sem objeto {key!r}.")
    return value


def _required_number(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Política de coleta sem número {key!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Número inválido para {key!r} na política de coleta.")
    return result


def _required_bool(config: dict[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Política de coleta sem booleano {key!r}.")
    return value


def load_collection_policy(path: str | Path) -> CollectionPolicy:
    config = load_yaml_config(path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão da política de coleta não suportada.")
    weight = _required_mapping(config, "weight_kg")
    min_weight = _required_number(weight, "min")
    max_weight = _required_number(weight, "max")
    if min_weight <= 0 or min_weight >= max_weight:
        raise ValueError("Intervalo de peso da política de coleta é inválido.")
    max_delta = config.get("max_capture_weight_delta_minutes")
    if isinstance(max_delta, bool) or not isinstance(max_delta, int) or max_delta <= 0:
        raise ValueError("max_capture_weight_delta_minutes deve ser inteiro positivo.")

    return CollectionPolicy(
        policy_id=_required_text(config, "policy_id"),
        policy_version=_required_text(config, "policy_version"),
        required_columns=_required_text_list(config, "required_columns"),
        allowed_views=_required_text_list(config, "allowed_views"),
        allowed_breeds=_required_text_list(config, "allowed_breeds"),
        allowed_sexes=_required_text_list(config, "allowed_sexes"),
        min_weight_kg=min_weight,
        max_weight_kg=max_weight,
        max_capture_weight_delta_minutes=max_delta,
        required_quality=_required_text(config, "required_quality"),
        require_scale_marker=_required_bool(config, "require_scale_marker"),
        require_commercial_training_rights=_required_bool(
            config, "require_commercial_training_rights"
        ),
    )


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} deve ser true ou false")


def _parse_date(value: str, *, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field} deve usar o formato YYYY-MM-DD") from exc


def _parse_timestamp(value: str, *, field: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} deve usar ISO 8601 com fuso horário") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} deve incluir o fuso horário")
    return result


def read_authorization_registry(
    path: str | Path,
) -> dict[str, AuthorizationRecord]:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Registro de autorizações não encontrado: {path}")
    with registry_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("O registro de autorizações não possui cabeçalho.")
        missing = set(AUTHORIZATION_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "Colunas de autorização ausentes: " + ", ".join(sorted(missing))
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]
    if not rows:
        raise ValueError("O registro de autorizações está vazio.")

    records: dict[str, AuthorizationRecord] = {}
    errors: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        authorization_id = row["authorization_id"]
        if not authorization_id:
            errors.append(f"linha {line_number}: authorization_id vazio")
            continue
        if authorization_id in records:
            errors.append(
                f"linha {line_number}: authorization_id duplicado: {authorization_id}"
            )
            continue
        try:
            effective_from = _parse_date(
                row["effective_from"], field="effective_from"
            )
            effective_until = (
                _parse_date(row["effective_until"], field="effective_until")
                if row["effective_until"]
                else None
            )
            if effective_until is not None and effective_until < effective_from:
                raise ValueError("effective_until é anterior a effective_from")
            records[authorization_id] = AuthorizationRecord(
                authorization_id=authorization_id,
                farm_id=row["farm_id"],
                status=row["status"],
                effective_from=effective_from,
                effective_until=effective_until,
                allows_model_training=_parse_bool(
                    row["allows_model_training"], field="allows_model_training"
                ),
                allows_commercial_use=_parse_bool(
                    row["allows_commercial_use"], field="allows_commercial_use"
                ),
                allows_data_sharing=_parse_bool(
                    row["allows_data_sharing"], field="allows_data_sharing"
                ),
                document_reference=row["document_reference"],
            )
            if not row["farm_id"] or not row["status"] or not row["document_reference"]:
                raise ValueError(
                    "farm_id, status e document_reference são obrigatórios"
                )
        except ValueError as exc:
            records.pop(authorization_id, None)
            errors.append(f"linha {line_number}: {exc}")
    if errors:
        raise ValueError("Registro de autorizações inválido:\n- " + "\n- ".join(errors))
    return records


def audit_pilot_collection(
    rows: list[dict[str, str]],
    authorizations: dict[str, AuthorizationRecord],
    policy: CollectionPolicy,
    *,
    manifest_path: str | Path,
    image_root: str | Path | None,
    check_images: bool,
    quality_policy: ImageQualityPolicy | None = None,
) -> CollectionAuditReport:
    errors: list[str] = []
    warnings: list[str] = []
    headers = set(rows[0]) if rows else set()
    missing_columns = set(policy.required_columns) - headers
    if missing_columns:
        errors.append(
            "Colunas da coleta ausentes: " + ", ".join(sorted(missing_columns))
        )
    try:
        validate_rows(
            rows,
            manifest_path=manifest_path,
            image_root=image_root,
            check_images=check_images,
        )
    except ValueError as exc:
        errors.append(str(exc))

    animal_farms: dict[str, str] = {}
    event_owners: dict[str, tuple[str, str]] = {}
    event_views: set[tuple[str, str]] = set()
    technical_quality_passed = 0
    for line_number, row in enumerate(rows, start=2):
        prefix = f"linha {line_number}"
        view = row.get("view", "")
        breed = row.get("breed", "")
        sex = row.get("sex", "")
        if view not in policy.allowed_views:
            errors.append(f"{prefix}: view não permitido: {view!r}")
        if breed not in policy.allowed_breeds:
            errors.append(f"{prefix}: breed não permitido: {breed!r}")
        if sex not in policy.allowed_sexes:
            errors.append(f"{prefix}: sex não permitido: {sex!r}")
        if row.get("quality", "") != policy.required_quality:
            errors.append(
                f"{prefix}: quality deve ser {policy.required_quality!r}"
            )

        try:
            weight = float(row.get("weight_kg", ""))
            if not policy.min_weight_kg <= weight <= policy.max_weight_kg:
                errors.append(
                    f"{prefix}: weight_kg fora de "
                    f"{policy.min_weight_kg:.0f}–{policy.max_weight_kg:.0f} kg"
                )
        except ValueError:
            pass

        try:
            has_marker = _parse_bool(
                row.get("scale_marker", ""), field="scale_marker"
            )
            if policy.require_scale_marker and not has_marker:
                errors.append(f"{prefix}: marcador de escala é obrigatório")
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
        try:
            commercial_training_allowed = _parse_bool(
                row.get("commercial_training_allowed", ""),
                field="commercial_training_allowed",
            )
            if (
                policy.require_commercial_training_rights
                and not commercial_training_allowed
            ):
                errors.append(
                    f"{prefix}: registro não autoriza treinamento comercial"
                )
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

        captured_at: datetime | None = None
        try:
            captured_at = _parse_timestamp(
                row.get("captured_at", ""), field="captured_at"
            )
            weighed_at = _parse_timestamp(
                row.get("weighed_at", ""), field="weighed_at"
            )
            delta_minutes = abs((captured_at - weighed_at).total_seconds()) / 60
            if delta_minutes > policy.max_capture_weight_delta_minutes:
                errors.append(
                    f"{prefix}: foto e pesagem estão separadas por "
                    f"{delta_minutes:.1f} minutos"
                )
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")

        animal_id = row.get("animal_id", "")
        event_id = row.get("event_id", "")
        farm_id = row.get("farm_id", "")
        previous_farm = animal_farms.setdefault(animal_id, farm_id)
        if previous_farm != farm_id:
            errors.append(f"{prefix}: animal aparece em fazendas diferentes")
        previous_owner = event_owners.setdefault(event_id, (animal_id, farm_id))
        if previous_owner != (animal_id, farm_id):
            errors.append(f"{prefix}: event_id reutilizado por outro animal ou fazenda")
        event_view = (event_id, view)
        if event_view in event_views:
            errors.append(f"{prefix}: mais de uma imagem selecionada para evento/vista")
        event_views.add(event_view)

        authorization_id = row.get("authorization_id", "")
        authorization = authorizations.get(authorization_id)
        if authorization is None:
            errors.append(f"{prefix}: authorization_id inexistente")
        else:
            if authorization.farm_id != farm_id:
                errors.append(f"{prefix}: autorização pertence a outra fazenda")
            if authorization.status != "approved":
                errors.append(f"{prefix}: autorização não está aprovada")
            if policy.require_commercial_training_rights and (
                not authorization.allows_model_training
                or not authorization.allows_commercial_use
            ):
                errors.append(
                    f"{prefix}: autorização não cobre treinamento e uso comercial"
                )
            if captured_at is not None:
                capture_date = captured_at.date()
                if capture_date < authorization.effective_from or (
                    authorization.effective_until is not None
                    and capture_date > authorization.effective_until
                ):
                    errors.append(f"{prefix}: captura fora da vigência da autorização")

        if check_images and quality_policy is not None:
            image_path = resolve_image_path(row, manifest_path, image_root)
            if image_path.is_file():
                try:
                    quality_report = assess_image_quality(image_path, quality_policy)
                    if quality_report.accepted:
                        technical_quality_passed += 1
                    else:
                        reasons = "; ".join(quality_report.rejection_reasons)
                        errors.append(
                            f"{prefix}: qualidade técnica rejeitada: {reasons}"
                        )
                except ValueError as exc:
                    errors.append(f"{prefix}: {exc}")

    if not check_images:
        warnings.append("Arquivos e qualidade técnica não foram verificados.")
    elif quality_policy is None:
        warnings.append("Política de qualidade técnica não foi aplicada.")

    return CollectionAuditReport(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        total_images=len(rows),
        animals=len({row.get("animal_id", "") for row in rows}),
        events=len({row.get("event_id", "") for row in rows}),
        farms=len({row.get("farm_id", "") for row in rows}),
        technical_quality_passed=technical_quality_passed,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
