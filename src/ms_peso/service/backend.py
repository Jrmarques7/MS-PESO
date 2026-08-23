from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from ms_peso.commercial_inference import (
    CommercialCandidatePredictor,
    CommercialWeightPrediction,
)
from ms_peso.commercial_model_package import (
    CommercialCandidateDescriptor,
    load_commercial_candidate_descriptor,
    verify_commercial_quality_policy,
)
from ms_peso.config import load_yaml_config
from ms_peso.image_quality import (
    ImageQualityPolicy,
    ImageQualityReport,
    assess_image_quality,
)
from ms_peso.service.config import ServiceSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendStatus:
    ready: bool
    code: str
    detail: str
    model: dict[str, object] | None = None


@dataclass(frozen=True)
class BackendPrediction:
    prediction: CommercialWeightPrediction | None
    quality: ImageQualityReport
    descriptor: CommercialCandidateDescriptor
    device: str | None


class PredictionBackend(Protocol):
    @property
    def status(self) -> BackendStatus: ...

    def initialize(self) -> None: ...

    def predict(self, image_path: Path) -> BackendPrediction: ...


def _public_model_summary(config: dict[str, object]) -> dict[str, object]:
    return {
        "id": config.get("model_id"),
        "version": config.get("model_version"),
        "status": config.get("status"),
        "production_ready": config.get("production_ready"),
        "commercial_use_allowed": config.get("commercial_use_allowed"),
    }


class CandidatePackageBackend:
    """Adapter for the current, explicitly unapproved candidate package.

    Serving it is disabled by default. The opt-in exists only so consumers can
    validate the HTTP contract in an isolated, non-commercial environment.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._status = BackendStatus(
            ready=False,
            code="not_initialized",
            detail="O pacote de inferência ainda não foi inicializado.",
        )
        self._descriptor: CommercialCandidateDescriptor | None = None
        self._quality_policy: ImageQualityPolicy | None = None
        self._predictor: CommercialCandidatePredictor | None = None
        self._inference_lock = Lock()

    @property
    def status(self) -> BackendStatus:
        return self._status

    def initialize(self) -> None:
        try:
            raw_config = load_yaml_config(self.settings.package_path)
        except (OSError, ValueError) as exc:
            logger.error("Invalid MS-PESO package descriptor: %s", exc)
            self._status = BackendStatus(
                ready=False,
                code="package_invalid",
                detail="O descritor do modelo está ausente ou inválido.",
            )
            return

        model_summary = _public_model_summary(raw_config)
        is_approved = (
            raw_config.get("production_ready") is True
            and raw_config.get("commercial_use_allowed") is True
        )
        if is_approved:
            self._status = BackendStatus(
                ready=False,
                code="production_package_not_supported",
                detail=(
                    "O carregador do pacote promovido ainda precisa ser ligado ao "
                    "serviço após a aprovação final."
                ),
                model=model_summary,
            )
            return
        if not self.settings.allow_unapproved_candidate:
            self._status = BackendStatus(
                ready=False,
                code="model_not_promoted",
                detail=(
                    "O candidato não está aprovado para produção ou uso comercial."
                ),
                model=model_summary,
            )
            return

        try:
            descriptor = load_commercial_candidate_descriptor(
                self.settings.package_path
            )
            quality_policy = verify_commercial_quality_policy(descriptor)
            predictor = CommercialCandidatePredictor.load(
                descriptor, device=self.settings.device
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Unable to initialize MS-PESO candidate: %s", exc)
            self._status = BackendStatus(
                ready=False,
                code="package_unavailable",
                detail="Os artefatos verificados do candidato não estão disponíveis.",
                model=model_summary,
            )
            return

        self._descriptor = descriptor
        self._quality_policy = quality_policy
        self._predictor = predictor
        self._status = BackendStatus(
            ready=True,
            code="research_candidate_ready",
            detail=(
                "Candidato interno disponível apenas para teste isolado de integração."
            ),
            model=model_summary,
        )

    def predict(self, image_path: Path) -> BackendPrediction:
        if not self.status.ready:
            raise RuntimeError("Backend de inferência indisponível.")
        if (
            self._descriptor is None
            or self._quality_policy is None
            or self._predictor is None
        ):
            raise RuntimeError("Backend inconsistente após inicialização.")

        quality = assess_image_quality(image_path, self._quality_policy)
        if not quality.accepted:
            return BackendPrediction(
                prediction=None,
                quality=quality,
                descriptor=self._descriptor,
                device=None,
            )
        with self._inference_lock:
            prediction = self._predictor.predict_image(image_path)
        return BackendPrediction(
            prediction=prediction,
            quality=quality,
            descriptor=self._descriptor,
            device=str(self._predictor.device),
        )
