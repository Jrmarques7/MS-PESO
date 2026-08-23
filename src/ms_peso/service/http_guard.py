from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestBodyTooLarge(Exception):
    pass


def _problem(code: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "detail": detail}},
    )


class RequestGuardMiddleware:
    """Reject unauthorized or oversized inference requests before parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str,
        authentication_configured: bool,
        max_prediction_body_bytes: int,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.authentication_configured = authentication_configured
        self.max_prediction_body_bytes = max_prediction_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        if path.startswith("/v1/"):
            if not self.authentication_configured:
                response = _problem(
                    "authentication_not_configured",
                    "O serviço não possui uma chave de API válida configurada.",
                    503,
                )
                await response(scope, receive, send)
                return
            supplied_key = headers.get(b"x-api-key", b"").decode(
                "utf-8", errors="replace"
            )
            if not hmac.compare_digest(supplied_key, self.api_key):
                response = _problem(
                    "unauthorized", "Chave de API ausente ou inválida.", 401
                )
                await response(scope, receive, send)
                return

        is_prediction = path == "/v1/predictions" and scope.get("method") == "POST"
        if not is_prediction:
            await self.app(scope, receive, send)
            return

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                response = _problem(
                    "invalid_content_length",
                    "O tamanho declarado da requisição é inválido.",
                    400,
                )
                await response(scope, receive, send)
                return
            if declared_size > self.max_prediction_body_bytes:
                response = _problem(
                    "request_too_large",
                    "A requisição excede o limite permitido.",
                    413,
                )
                await response(scope, receive, send)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_prediction_body_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            response = _problem(
                "request_too_large",
                "A requisição excede o limite permitido.",
                413,
            )
            await response(scope, receive, send)
