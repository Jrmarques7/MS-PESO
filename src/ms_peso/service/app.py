from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse

from ms_peso.service.backend import CandidatePackageBackend, PredictionBackend
from ms_peso.service.config import ServiceSettings
from ms_peso.service.http_guard import RequestGuardMiddleware
from ms_peso.service.payloads import build_prediction_payload
from ms_peso.service.uploads import UploadValidationError, store_image_upload

logger = logging.getLogger(__name__)


class Unauthorized(Exception):
    pass


class ServiceUnavailable(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _problem(code: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "detail": detail}},
    )


def create_app(
    *,
    settings: ServiceSettings | None = None,
    backend: PredictionBackend | None = None,
) -> FastAPI:
    service_settings = settings or ServiceSettings.from_env()
    prediction_backend = backend or CandidatePackageBackend(service_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        prediction_backend.initialize()
        yield

    app = FastAPI(
        title="MS-PESO Inference API",
        version="1.0.0",
        description=(
            "Serviço stateless de estimativa de peso bovino por imagem. "
            "O candidato atual permanece bloqueado para uso comercial."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestGuardMiddleware,
        api_key=service_settings.api_key,
        authentication_configured=service_settings.authentication_configured,
        # Multipart adds boundaries and headers around the image itself. The
        # image still has its exact, lower limit enforced while it is copied.
        max_prediction_body_bytes=(service_settings.max_upload_bytes + 1024 * 1024),
    )

    def require_api_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if not service_settings.authentication_configured:
            raise ServiceUnavailable(
                "authentication_not_configured",
                "O serviço não possui uma chave de API válida configurada.",
            )
        if x_api_key is None or not hmac.compare_digest(
            x_api_key, service_settings.api_key
        ):
            raise Unauthorized("Chave de API ausente ou inválida.")

    @app.exception_handler(Unauthorized)
    async def unauthorized_handler(_, exc: Unauthorized) -> JSONResponse:
        return _problem("unauthorized", str(exc), 401)

    @app.exception_handler(ServiceUnavailable)
    async def unavailable_handler(_, exc: ServiceUnavailable) -> JSONResponse:
        return _problem(exc.code, str(exc), 503)

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, object]:
        return {"status": "ok", "service": "ms-peso"}

    @app.get("/health/ready", tags=["health"])
    def ready() -> JSONResponse:
        status = prediction_backend.status
        payload = {
            "status": "ready" if status.ready else "not_ready",
            "code": status.code,
            "detail": status.detail,
            "model": status.model,
            "authentication_configured": (service_settings.authentication_configured),
        }
        is_ready = status.ready and service_settings.authentication_configured
        if not service_settings.authentication_configured:
            payload["status"] = "not_ready"
            payload["code"] = "authentication_not_configured"
            payload["detail"] = "Configure uma chave de API com ao menos 32 caracteres."
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)

    @app.get(
        "/v1/model",
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
    )
    def model_status() -> JSONResponse:
        status = prediction_backend.status
        return JSONResponse(
            status_code=200 if status.ready else 503,
            content={
                "ready": status.ready,
                "code": status.code,
                "detail": status.detail,
                "model": status.model,
            },
        )

    @app.post(
        "/v1/predictions",
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
    )
    async def predict(
        image: Annotated[UploadFile, File(description="Imagem lateral do bovino")],
        correlation_id: Annotated[str | None, Form(max_length=128)] = None,
    ) -> JSONResponse:
        backend_status = prediction_backend.status
        if not backend_status.ready:
            return _problem(backend_status.code, backend_status.detail, 503)

        try:
            upload = await store_image_upload(
                image, max_bytes=service_settings.max_upload_bytes
            )
        except UploadValidationError as exc:
            return _problem(exc.code, exc.detail, exc.status_code)

        try:
            result = prediction_backend.predict(upload.path)
            payload = build_prediction_payload(
                result, upload, correlation_id=correlation_id
            )
            status_code = 200 if result.prediction is not None else 422
            return JSONResponse(status_code=status_code, content=payload)
        except ValueError:
            return _problem(
                "invalid_image",
                "O arquivo enviado não contém uma imagem válida.",
                422,
            )
        except RuntimeError:
            logger.exception("MS-PESO inference failed")
            return _problem(
                "inference_failed",
                "A inferência não pôde ser concluída.",
                500,
            )
        finally:
            upload.remove()

    return app


app = create_app()
