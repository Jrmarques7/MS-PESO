from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ms_peso.config import load_yaml_config
from ms_peso.conformal import validate_target_coverage
from ms_peso.image_quality import ImageQualityPolicy, load_image_quality_policy
from ms_peso.integrity import calculate_sha256

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CommercialCandidateDescriptor:
    model_id: str
    model_version: str
    status: str
    production_ready: bool
    commercial_use_allowed: bool
    commercial_blockers: tuple[str, ...]
    architecture: str
    checkpoint_path: Path
    checkpoint_sha256: str
    calibration_path: Path
    calibration_sha256: str
    evaluation_path: Path
    evaluation_sha256: str
    quality_policy_path: Path
    quality_policy_sha256: str
    image_size: int
    input_view: str
    dataset: str
    breed: str
    limitations: tuple[str, ...]
    model_card_path: Path
    model_card_sha256: str


@dataclass(frozen=True)
class VerifiedCommercialCandidate:
    checkpoint: dict[str, Any]
    target_coverage: float
    interval_radius_kg: float
    evaluation: dict[str, Any]
    quality_policy: ImageQualityPolicy


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Descritor comercial sem objeto {key!r}.")
    return value


def _required_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Descritor comercial sem texto {key!r}.")
    return value.strip()


def _required_sha256(config: dict[str, Any], key: str) -> str:
    value = _required_text(config, key).lower()
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"SHA-256 inválido em {key!r}.")
    return value


def _required_string_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{key} deve ser uma lista de textos não vazia.")
    return tuple(item.strip() for item in value)


def _artifact_path(
    descriptor_path: Path, config: dict[str, Any]
) -> tuple[Path, str]:
    return (
        (descriptor_path.parent / _required_text(config, "path")).resolve(),
        _required_sha256(config, "sha256"),
    )


def load_commercial_candidate_descriptor(
    path: str | Path,
) -> CommercialCandidateDescriptor:
    """Carrega somente o contrato de um candidato ainda não autorizado."""
    descriptor_path = Path(path).resolve()
    config = load_yaml_config(descriptor_path)
    if config.get("schema_version") != 1:
        raise ValueError("Versão do descritor comercial não suportada.")
    status = _required_text(config, "status")
    if status != "candidate_unapproved":
        raise ValueError("O descritor deve permanecer candidate_unapproved.")
    production_ready = config.get("production_ready")
    commercial_use_allowed = config.get("commercial_use_allowed")
    if production_ready is not False or commercial_use_allowed is not False:
        raise ValueError("O candidato interno não pode declarar liberação comercial.")
    commercial_blockers = _required_string_list(config, "commercial_blockers")

    checkpoint_path, checkpoint_sha256 = _artifact_path(
        descriptor_path, _required_mapping(config, "checkpoint")
    )
    calibration_path, calibration_sha256 = _artifact_path(
        descriptor_path, _required_mapping(config, "calibration")
    )
    evaluation_path, evaluation_sha256 = _artifact_path(
        descriptor_path, _required_mapping(config, "evaluation")
    )
    quality_policy_path, quality_policy_sha256 = _artifact_path(
        descriptor_path, _required_mapping(config, "quality_policy")
    )
    input_config = _required_mapping(config, "input")
    image_size = input_config.get("image_size")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("input.image_size deve ser um inteiro positivo.")
    domain = _required_mapping(config, "domain")
    model_card_path, model_card_sha256 = _artifact_path(
        descriptor_path, _required_mapping(config, "model_card")
    )
    return CommercialCandidateDescriptor(
        model_id=_required_text(config, "model_id"),
        model_version=_required_text(config, "model_version"),
        status=status,
        production_ready=production_ready,
        commercial_use_allowed=commercial_use_allowed,
        commercial_blockers=commercial_blockers,
        architecture=_required_text(config, "architecture"),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha256,
        evaluation_path=evaluation_path,
        evaluation_sha256=evaluation_sha256,
        quality_policy_path=quality_policy_path,
        quality_policy_sha256=quality_policy_sha256,
        image_size=image_size,
        input_view=_required_text(input_config, "view"),
        dataset=_required_text(domain, "dataset"),
        breed=_required_text(domain, "breed"),
        limitations=_required_string_list(config, "limitations"),
        model_card_path=model_card_path,
        model_card_sha256=model_card_sha256,
    )


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} não encontrado: {path}")
    actual_sha256 = calculate_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 de {label} diverge do descritor comercial.")


def verify_commercial_quality_policy(
    descriptor: CommercialCandidateDescriptor,
) -> ImageQualityPolicy:
    """Autentica a política antes de analisar a imagem recebida."""
    _verify_file(
        descriptor.quality_policy_path,
        descriptor.quality_policy_sha256,
        label="política de qualidade",
    )
    return load_image_quality_policy(descriptor.quality_policy_path)


def _load_json_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} não contém JSON válido.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} inválido.")
    return payload


def verify_commercial_candidate_package(
    descriptor: CommercialCandidateDescriptor,
) -> VerifiedCommercialCandidate:
    """Verifica toda a cadeia técnica sem conceder autorização comercial."""
    _verify_file(
        descriptor.checkpoint_path,
        descriptor.checkpoint_sha256,
        label="checkpoint",
    )
    _verify_file(
        descriptor.calibration_path,
        descriptor.calibration_sha256,
        label="calibração",
    )
    _verify_file(
        descriptor.evaluation_path,
        descriptor.evaluation_sha256,
        label="avaliação final",
    )
    quality_policy = verify_commercial_quality_policy(descriptor)
    _verify_file(
        descriptor.model_card_path,
        descriptor.model_card_sha256,
        label="model card comercial",
    )

    checkpoint = torch.load(
        descriptor.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint comercial inválido.")
    required_checkpoint_fields = {
        "model_state_dict",
        "architecture",
        "dropout",
        "target_mean",
        "target_std",
        "config",
        "workflow",
        "initialization",
        "commercial_use_allowed",
        "promotion_status",
        "source_snapshot_id",
        "source_manifest_sha256",
        "source_split_report_sha256",
    }
    missing = required_checkpoint_fields - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint comercial sem campos obrigatórios: "
            + ", ".join(sorted(missing))
        )
    if checkpoint["workflow"] != "commercial_fit":
        raise ValueError("Checkpoint não pertence ao ajuste comercial.")
    if checkpoint["initialization"] != "random":
        raise ValueError("Checkpoint não possui inicialização aleatória.")
    if checkpoint["commercial_use_allowed"] is not False:
        raise ValueError("Checkpoint possui autorização comercial indevida.")
    if checkpoint["promotion_status"] != "not_promoted":
        raise ValueError("Checkpoint possui estado de promoção incompatível.")
    if checkpoint["architecture"] != descriptor.architecture:
        raise ValueError("Arquitetura diverge entre checkpoint e descritor.")
    fit_config = checkpoint["config"]
    if not isinstance(fit_config, dict):
        raise ValueError("Checkpoint sem configuração interna válida.")
    fit_project = _required_mapping(fit_config, "project")
    fit_model = _required_mapping(fit_config, "model")
    fit_data = _required_mapping(fit_config, "data")
    if fit_project.get("workflow") != "commercial_fit":
        raise ValueError("Configuração interna não pertence ao ajuste comercial.")
    if fit_model.get("pretrained") is not False:
        raise ValueError("Configuração interna contém pesos pré-treinados.")
    if fit_model.get("initialization") != "random":
        raise ValueError("Configuração interna não declara inicialização aleatória.")
    if fit_data.get("image_size") != descriptor.image_size:
        raise ValueError("Tamanho de imagem diverge do ajuste comercial.")
    if fit_data.get("view") != descriptor.input_view:
        raise ValueError("Vista de entrada diverge do ajuste comercial.")

    calibration = _load_json_artifact(
        descriptor.calibration_path, label="Relatório de calibração"
    )
    calibration_requirements = {
        "status": "calibrated",
        "workflow": "commercial_calibration",
        "method": "split_conformal_grouped_absolute_residual",
        "grouping": "animal_id_max_absolute_residual",
        "architecture": descriptor.architecture,
        "image_size": descriptor.image_size,
        "input_view": descriptor.input_view,
        "test_evaluated": False,
        "promotion_status": "not_promoted",
        "commercial_use_allowed": False,
        "source_checkpoint_sha256": descriptor.checkpoint_sha256,
    }
    for key, expected in calibration_requirements.items():
        if calibration.get(key) != expected:
            raise ValueError(f"{key} diverge no relatório de calibração.")
    try:
        target_coverage = float(calibration["target_coverage"])
        interval_radius_kg = float(calibration["interval_radius_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Parâmetros conformais inválidos.") from exc
    validate_target_coverage(target_coverage)
    if not math.isfinite(interval_radius_kg) or interval_radius_kg < 0:
        raise ValueError("Raio conformal inválido.")

    evaluation = _load_json_artifact(
        descriptor.evaluation_path, label="Relatório de avaliação final"
    )
    evaluation_requirements = {
        "status": "evaluated",
        "workflow": "commercial_evaluation",
        "test_evaluated": True,
        "test_consumed": True,
        "architecture": descriptor.architecture,
        "image_size": descriptor.image_size,
        "input_view": descriptor.input_view,
        "technical_criteria_passed": True,
        "technical_recommendation": "technical_review_recommended",
        "promotion_status": "review_required",
        "commercial_use_allowed": False,
        "source_checkpoint_sha256": descriptor.checkpoint_sha256,
        "source_calibration_sha256": descriptor.calibration_sha256,
        "interval_method": "split_conformal_grouped_absolute_residual",
    }
    for key, expected in evaluation_requirements.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"{key} diverge no relatório de avaliação final.")
    try:
        evaluation_target_coverage = float(evaluation["interval_target_coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cobertura da avaliação final é inválida.") from exc
    if not math.isfinite(evaluation_target_coverage) or not math.isclose(
        evaluation_target_coverage, target_coverage
    ):
        raise ValueError("Cobertura conformal diverge entre calibração e avaliação.")
    evaluation_coverage = evaluation.get("interval_coverage")
    if not isinstance(evaluation_coverage, dict):
        raise ValueError("Cobertura da avaliação final é inválida.")
    try:
        evaluation_radius_kg = float(evaluation_coverage["radius_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Raio da avaliação final é inválido.") from exc
    if not math.isfinite(evaluation_radius_kg) or not math.isclose(
        evaluation_radius_kg, interval_radius_kg
    ):
        raise ValueError("Raio conformal diverge entre calibração e avaliação.")

    source_keys = (
        "source_snapshot_id",
        "source_manifest_sha256",
        "source_split_report_sha256",
    )
    for key in source_keys:
        expected = checkpoint[key]
        if calibration.get(key) != expected or evaluation.get(key) != expected:
            raise ValueError(f"{key} diverge na cadeia do candidato comercial.")
    remaining_reviews = evaluation.get("mandatory_remaining_reviews")
    required_reviews = {
        "legal_rights_review",
        "external_domain_validation",
        "operational_safety_review",
        "human_promotion_approval",
    }
    if not isinstance(remaining_reviews, list) or not required_reviews.issubset(
        set(remaining_reviews)
    ):
        raise ValueError("Avaliação não registra todas as revisões obrigatórias.")

    return VerifiedCommercialCandidate(
        checkpoint=checkpoint,
        target_coverage=target_coverage,
        interval_radius_kg=interval_radius_kg,
        evaluation=evaluation,
        quality_policy=quality_policy,
    )
