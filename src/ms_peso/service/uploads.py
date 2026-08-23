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
ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
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


async def _store_upload(
    upload: UploadFile,
    *,
    max_bytes: int,
    allowed_types: set[str],
    temporary_suffixes: dict[str, str],
    unsupported_detail: str,
    too_large_code: str,
    too_large_label: str,
    empty_code: str,
    empty_detail: str,
) -> StoredUpload:
    content_type = (upload.content_type or "").lower()
    if content_type not in allowed_types:
        raise UploadValidationError(
            "unsupported_media_type",
            unsupported_detail,
            status_code=415,
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="ms-peso-", suffix=temporary_suffixes.get(content_type, ".upload")
    )
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadValidationError(
                        too_large_code,
                        f"{too_large_label} excede o limite de {max_bytes} bytes.",
                        status_code=413,
                    )
                destination.write(chunk)
        if size == 0:
            raise UploadValidationError(
                empty_code,
                empty_detail,
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


async def store_image_upload(upload: UploadFile, *, max_bytes: int) -> StoredUpload:
    return await _store_upload(
        upload,
        max_bytes=max_bytes,
        allowed_types=ALLOWED_IMAGE_TYPES,
        temporary_suffixes={},
        unsupported_detail="Envie uma imagem JPEG, PNG ou WebP.",
        too_large_code="image_too_large",
        too_large_label="A imagem",
        empty_code="empty_image",
        empty_detail="O arquivo de imagem está vazio.",
    )


async def store_video_upload(upload: UploadFile, *, max_bytes: int) -> StoredUpload:
    return await _store_upload(
        upload,
        max_bytes=max_bytes,
        allowed_types=set(ALLOWED_VIDEO_TYPES),
        temporary_suffixes=ALLOWED_VIDEO_TYPES,
        unsupported_detail="Envie um vídeo MP4, MOV, WebM ou AVI.",
        too_large_code="video_too_large",
        too_large_label="O vídeo",
        empty_code="empty_video",
        empty_detail="O arquivo de vídeo está vazio.",
    )
