from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ms_peso.commercial_inference import CommercialWeightPrediction
from ms_peso.commercial_model_package import CommercialCandidateDescriptor
from ms_peso.image_quality import ImageQualityReport, QualityCheck
from ms_peso.service.app import create_app
from ms_peso.service.backend import (
    BackendPrediction,
    BackendStatus,
    CandidatePackageBackend,
)
from ms_peso.service.config import ServiceSettings

API_KEY = "integration-test-key-with-more-than-32-characters"


def _settings(tmp_path: Path, **overrides: object) -> ServiceSettings:
    values = {
        "api_key": API_KEY,
        "package_path": tmp_path / "candidate.yaml",
        "device": "cpu",
        "max_upload_bytes": 1024 * 1024,
        "allow_unapproved_candidate": False,
    }
    values.update(overrides)
    return ServiceSettings(**values)


def _descriptor(tmp_path: Path) -> CommercialCandidateDescriptor:
    return CommercialCandidateDescriptor(
        model_id="candidate-test",
        model_version="1",
        status="candidate_unapproved",
        production_ready=False,
        commercial_use_allowed=False,
        commercial_blockers=("Aprovação final pendente.",),
        architecture="efficientnet_b0",
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256="a" * 64,
        calibration_path=tmp_path / "calibration.json",
        calibration_sha256="b" * 64,
        evaluation_path=tmp_path / "evaluation.json",
        evaluation_sha256="c" * 64,
        quality_policy_path=tmp_path / "quality.yaml",
        quality_policy_sha256="d" * 64,
        image_size=224,
        input_view="left",
        dataset="owned-pilot",
        breed="nelore",
        limitations=("Somente teste interno.",),
        model_card_path=tmp_path / "model-card.md",
        model_card_sha256="e" * 64,
    )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (100, 120, 140)).save(buffer, format="PNG")
    return buffer.getvalue()


def _quality(*, accepted: bool) -> ImageQualityReport:
    check = QualityCheck(
        code="minimum_resolution",
        passed=accepted,
        value={"width": 64, "height": 48},
        requirement=">= 32x24",
        rejection_message="Resolução insuficiente.",
    )
    return ImageQualityReport(
        policy_id="quality-test",
        policy_version="1",
        width=64,
        height=48,
        checks=(check,),
        limitations=("Não valida pose.",),
    )


class FakeBackend:
    def __init__(self, tmp_path: Path, *, accepted: bool = True) -> None:
        self._status = BackendStatus(
            ready=True,
            code="research_candidate_ready",
            detail="Teste interno.",
            model={"id": "candidate-test"},
        )
        self.descriptor = _descriptor(tmp_path)
        self.accepted = accepted
        self.path_seen: Path | None = None

    @property
    def status(self) -> BackendStatus:
        return self._status

    def initialize(self) -> None:
        pass

    def predict(self, image_path: Path) -> BackendPrediction:
        assert image_path.is_file()
        self.path_seen = image_path
        prediction = None
        if self.accepted:
            prediction = CommercialWeightPrediction(
                estimated_weight_kg=410.0,
                interval_lower_kg=390.0,
                interval_upper_kg=430.0,
                interval_radius_kg=20.0,
                target_coverage=0.9,
                lower_bound_clipped_at_zero=False,
                original_width=64,
                original_height=48,
            )
        return BackendPrediction(
            prediction=prediction,
            quality=_quality(accepted=self.accepted),
            descriptor=self.descriptor,
            device="cpu" if prediction else None,
        )


def test_liveness_does_not_depend_on_model_or_authentication(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    settings = _settings(tmp_path, api_key="")
    with TestClient(create_app(settings=settings, backend=backend)) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ms-peso"}


def test_readiness_reports_authentication_configuration(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    settings = _settings(tmp_path, api_key="short")
    with TestClient(create_app(settings=settings, backend=backend)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["code"] == "authentication_not_configured"


def test_prediction_requires_valid_api_key(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    app = create_app(settings=_settings(tmp_path), backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("cow.png", _image_bytes(), "image/png")},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_prediction_returns_traceable_contract_and_removes_upload(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(tmp_path)
    app = create_app(settings=_settings(tmp_path), backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            headers={"X-API-Key": API_KEY},
            files={"image": ("cow.png", _image_bytes(), "image/png")},
            data={"correlation_id": "farmup-lot-42-animal-7"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["correlation_id"] == "farmup-lot-42-animal-7"
    assert payload["estimated_weight_kg"] == 410.0
    assert payload["prediction_interval"]["lower_kg"] == 390.0
    assert payload["model"]["commercial_use_allowed"] is False
    assert payload["authorization_status"] == "blocked_pending_mandatory_reviews"
    assert payload["input"]["filename"] == "cow.png"
    assert "path" not in payload["input"]
    assert backend.path_seen is not None
    assert not backend.path_seen.exists()


def test_quality_rejection_keeps_structured_contract(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path, accepted=False)
    app = create_app(settings=_settings(tmp_path), backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            headers={"X-API-Key": API_KEY},
            files={"image": ("cow.png", _image_bytes(), "image/png")},
        )
    assert response.status_code == 422
    payload = response.json()
    assert payload["prediction_status"] == "rejected"
    assert payload["estimated_weight_kg"] is None
    assert payload["quality"]["rejection_reasons"] == ["Resolução insuficiente."]


def test_rejects_unsupported_or_large_upload(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    settings = _settings(tmp_path, max_upload_bytes=10)
    with TestClient(create_app(settings=settings, backend=backend)) as client:
        unsupported = client.post(
            "/v1/predictions",
            headers={"X-API-Key": API_KEY},
            files={"image": ("cow.gif", b"GIF89a", "image/gif")},
        )
        too_large = client.post(
            "/v1/predictions",
            headers={"X-API-Key": API_KEY},
            files={"image": ("cow.png", _image_bytes(), "image/png")},
        )
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "unsupported_media_type"
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "image_too_large"


def test_rejects_declared_oversized_body_before_multipart_parsing(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(tmp_path)
    settings = _settings(tmp_path, max_upload_bytes=1024)
    app = create_app(settings=settings, backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "multipart/form-data; boundary=test",
                "Content-Length": str(2 * 1024 * 1024),
            },
            content=b"not parsed",
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert backend.path_seen is None


def test_default_backend_blocks_unapproved_descriptor(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "candidate.yaml"
    descriptor_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model_id: candidate-test",
                'model_version: "1"',
                "status: candidate_unapproved",
                "production_ready: false",
                "commercial_use_allowed: false",
            ]
        ),
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    backend = CandidatePackageBackend(settings)
    with TestClient(create_app(settings=settings, backend=backend)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["code"] == "model_not_promoted"
