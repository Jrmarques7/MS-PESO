from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class UploadValidationError(ValueError):
    def __init__(self, code: str, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    size_bytes: int
    original_filename: str
    content_type: str

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


async def store_image_upload(upload: UploadFile, *, max_bytes: int) -> StoredUpload:
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise UploadValidationError(
            "unsupported_media_type",
            "Envie uma imagem JPEG, PNG ou WebP.",
            status_code=415,
        )

    descriptor, temporary_name = tempfile.mkstemp(prefix="ms-peso-", suffix=".img")
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadValidationError(
                        "image_too_large",
                        f"A imagem excede o limite de {max_bytes} bytes.",
                        status_code=413,
                    )
                destination.write(chunk)
        if size == 0:
            raise UploadValidationError(
                "empty_image",
                "O arquivo de imagem está vazio.",
                status_code=422,
            )
        return StoredUpload(
            path=Path(temporary_name),
            size_bytes=size,
            original_filename=Path(upload.filename or "image").name,
            content_type=content_type,
        )
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
