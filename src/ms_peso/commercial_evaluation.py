from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ms_peso.commercial_checkpoint import (
    CommercialFitCheckpoint,
    load_commercial_fit_checkpoint,
)
from ms_peso.commercial_data import CommercialDataContract
from ms_peso.conformal import validate_target_coverage
from ms_peso.integrity import calculate_sha256


@dataclass(frozen=True)
class AcceptanceCriteria:
    minimum_test_animals: int
    maximum_mae_upper_95_kg: float
    maximum_rmse_upper_95_kg: float
    maximum_mape_upper_95_pct: float
    maximum_absolute_bias_bound_95_kg: float
    minimum_group_coverage_lower_95: float
    maximum_interval_radius_kg: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationArtifact:
    path: Path
    sha256: str
    target_coverage: float
    interval_radius_kg: float


@dataclass(frozen=True)
class CommercialEvaluationContract:
    checkpoint: CommercialFitCheckpoint
    calibration: CalibrationArtifact
    acceptance: AcceptanceCriteria
    bootstrap_iterations: int
    access_receipt_path: Path


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuração {key} ausente ou inválida.")
    return value


def _required_positive_float(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} deve ser numérico e positivo.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} deve ser numérico e positivo.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{key} deve ser numérico e positivo.")
    return result


def _required_sha256(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value.lower()):
        raise ValueError(f"{key} deve conter um SHA-256 válido.")
    return value.lower()


def _load_calibration_artifact(
    calibration_config: dict[str, Any],
    *,
    checkpoint: CommercialFitCheckpoint,
    data_contract: CommercialDataContract,
) -> CalibrationArtifact:
    path_value = calibration_config.get("report")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("calibration.report deve apontar para o relatório congelado.")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Relatório de calibração não encontrado: {path}")
    expected_sha256 = _required_sha256(calibration_config, "report_sha256")
    actual_sha256 = calculate_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Hash do relatório de calibração diverge da configuração.")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Relatório de calibração não contém JSON válido.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Relatório de calibração inválido.")
    if payload.get("status") != "calibrated":
        raise ValueError("Relatório de calibração não está aprovado.")
    if payload.get("workflow") != "commercial_calibration":
        raise ValueError("Relatório não pertence à calibração comercial.")
    if payload.get("method") != "split_conformal_grouped_absolute_residual":
        raise ValueError("Método de calibração não suportado.")
    if payload.get("grouping") != "animal_id_max_absolute_residual":
        raise ValueError("Agrupamento de calibração não suportado.")
    if payload.get("test_evaluated") is not False:
        raise ValueError("O relatório indica consulta anterior ao teste.")
    if payload.get("promotion_status") != "not_promoted":
        raise ValueError("A calibração já possui estado de promoção.")
    if payload.get("commercial_use_allowed") is not False:
        raise ValueError("A calibração possui autorização comercial indevida.")

    provenance_checks = {
        "source_checkpoint_sha256": checkpoint.sha256,
        "source_snapshot_id": data_contract.snapshot_id,
        "source_manifest_sha256": data_contract.manifest_sha256,
        "source_split_report_sha256": data_contract.split_report_sha256,
        "architecture": checkpoint.architecture,
        "image_size": checkpoint.image_size,
        "input_view": checkpoint.input_view,
    }
    for key, expected in provenance_checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"{key} diverge no relatório de calibração.")

    try:
        target_coverage = float(payload["target_coverage"])
        interval_radius_kg = float(payload["interval_radius_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Parâmetros do intervalo calibrado são inválidos.") from exc
    validate_target_coverage(target_coverage)
    if not math.isfinite(interval_radius_kg) or interval_radius_kg < 0:
        raise ValueError("Raio do intervalo calibrado é inválido.")
    number_of_animals = payload.get("number_of_calibration_animals")
    quantile_rank = payload.get("quantile_rank")
    if (
        isinstance(number_of_animals, bool)
        or not isinstance(number_of_animals, int)
        or number_of_animals <= 0
        or isinstance(quantile_rank, bool)
        or not isinstance(quantile_rank, int)
        or not 1 <= quantile_rank <= number_of_animals
    ):
        raise ValueError("Tamanho ou quantil da calibração é inválido.")
    return CalibrationArtifact(
        path=path,
        sha256=actual_sha256,
        target_coverage=target_coverage,
        interval_radius_kg=interval_radius_kg,
    )


def _load_acceptance_criteria(config: dict[str, Any]) -> AcceptanceCriteria:
    minimum_test_animals = config.get("minimum_test_animals")
    if (
        isinstance(minimum_test_animals, bool)
        or not isinstance(minimum_test_animals, int)
        or minimum_test_animals <= 0
    ):
        raise ValueError("minimum_test_animals deve ser um inteiro positivo.")
    minimum_coverage = config.get("minimum_group_coverage_lower_95")
    if isinstance(minimum_coverage, bool):
        raise ValueError("minimum_group_coverage_lower_95 deve estar entre 0 e 1.")
    try:
        minimum_coverage = float(minimum_coverage)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "minimum_group_coverage_lower_95 deve estar entre 0 e 1."
        ) from exc
    if not math.isfinite(minimum_coverage) or not 0 < minimum_coverage < 1:
        raise ValueError("minimum_group_coverage_lower_95 deve estar entre 0 e 1.")
    return AcceptanceCriteria(
        minimum_test_animals=minimum_test_animals,
        maximum_mae_upper_95_kg=_required_positive_float(
            config, "maximum_mae_upper_95_kg"
        ),
        maximum_rmse_upper_95_kg=_required_positive_float(
            config, "maximum_rmse_upper_95_kg"
        ),
        maximum_mape_upper_95_pct=_required_positive_float(
            config, "maximum_mape_upper_95_pct"
        ),
        maximum_absolute_bias_bound_95_kg=_required_positive_float(
            config, "maximum_absolute_bias_bound_95_kg"
        ),
        minimum_group_coverage_lower_95=minimum_coverage,
        maximum_interval_radius_kg=_required_positive_float(
            config, "maximum_interval_radius_kg"
        ),
    )


def validate_commercial_evaluation_contract(
    config: dict[str, Any],
    data_contract: CommercialDataContract,
    *,
    output_dir: str | Path,
) -> CommercialEvaluationContract:
    """Congela as entradas e critérios antes de qualquer acesso ao teste."""
    project_config = _required_mapping(config, "project")
    if project_config.get("workflow") != "commercial_evaluation":
        raise ValueError(
            "A avaliação exige project.workflow: commercial_evaluation."
        )
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            "A saída da avaliação comercial já existe; escolha uma nova versão."
        )

    data_config = _required_mapping(config, "data")
    model_config = _required_mapping(config, "model")
    checkpoint_value = model_config.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError("A avaliação exige model.checkpoint.")
    checkpoint = load_commercial_fit_checkpoint(
        checkpoint_value,
        expected_sha256=_required_sha256(model_config, "checkpoint_sha256"),
        data_contract=data_contract,
        image_size=data_config.get("image_size"),
        input_view=data_config.get("view"),
    )
    calibration = _load_calibration_artifact(
        _required_mapping(config, "calibration"),
        checkpoint=checkpoint,
        data_contract=data_contract,
    )
    acceptance = _load_acceptance_criteria(_required_mapping(config, "acceptance"))
    if acceptance.minimum_group_coverage_lower_95 > calibration.target_coverage:
        raise ValueError(
            "A cobertura inferior exigida não pode superar a cobertura calibrada."
        )
    if calibration.interval_radius_kg > acceptance.maximum_interval_radius_kg:
        raise ValueError(
            "O intervalo calibrado excede o limite congelado; teste preservado."
        )
    if data_contract.number_of_animals["test"] < acceptance.minimum_test_animals:
        raise ValueError(
            "Animais de teste insuficientes para o critério congelado; teste "
            "preservado."
        )

    evaluation_config = _required_mapping(config, "evaluation")
    bootstrap_iterations = evaluation_config.get("bootstrap_iterations")
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, int)
        or bootstrap_iterations < 100
    ):
        raise ValueError("evaluation.bootstrap_iterations deve ser inteiro >= 100.")
    receipt_value = evaluation_config.get("access_receipt")
    if not isinstance(receipt_value, str) or not receipt_value.strip():
        raise ValueError("evaluation.access_receipt deve definir o recibo do teste.")
    receipt_path = Path(receipt_value)
    if receipt_path.exists():
        raise FileExistsError(
            "O teste já possui recibo de acesso e não pode ser consultado novamente."
        )
    receipt_inside_output = False
    try:
        receipt_path.resolve().relative_to(output_path.resolve())
        receipt_inside_output = True
    except ValueError:
        pass
    if receipt_inside_output:
        raise ValueError(
            "O recibo de acesso deve ficar fora do diretório de resultados."
        )
    return CommercialEvaluationContract(
        checkpoint=checkpoint,
        calibration=calibration,
        acceptance=acceptance,
        bootstrap_iterations=bootstrap_iterations,
        access_receipt_path=receipt_path,
    )
