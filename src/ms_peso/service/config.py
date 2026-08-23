from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} deve ser true ou false.")


def _read_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um número inteiro.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} deve estar entre {minimum} e {maximum}.")
    return value


@dataclass(frozen=True)
class ServiceSettings:
    """Configuration read by the HTTP adapter, never by the ML core."""

    api_key: str
    package_path: Path
    video_policy_path: Path = PROJECT_ROOT / "configs" / "video_inference.yaml"
    device: str = "auto"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_video_upload_bytes: int = 50 * 1024 * 1024
    allow_unapproved_candidate: bool = False
    host: str = "127.0.0.1"
    port: int = 8080

    @property
    def authentication_configured(self) -> bool:
        return len(self.api_key) >= 32

    @classmethod
    def from_env(cls) -> ServiceSettings:
        package_value = os.getenv(
            "MS_PESO_PACKAGE_PATH",
            str(PROJECT_ROOT / "models" / "commercial_candidate.yaml"),
        )
        video_policy_value = os.getenv(
            "MS_PESO_VIDEO_POLICY_PATH",
            str(PROJECT_ROOT / "configs" / "video_inference.yaml"),
        )
        device = os.getenv("MS_PESO_DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("MS_PESO_DEVICE deve ser auto, cpu ou cuda.")
        return cls(
            api_key=os.getenv("MS_PESO_API_KEY", ""),
            package_path=Path(package_value).expanduser().resolve(),
            video_policy_path=Path(video_policy_value).expanduser().resolve(),
            device=device,
            max_upload_bytes=_read_int(
                "MS_PESO_MAX_UPLOAD_BYTES",
                default=10 * 1024 * 1024,
                minimum=1024,
                maximum=50 * 1024 * 1024,
            ),
            max_video_upload_bytes=_read_int(
                "MS_PESO_MAX_VIDEO_UPLOAD_BYTES",
                default=50 * 1024 * 1024,
                minimum=1024,
                maximum=250 * 1024 * 1024,
            ),
            allow_unapproved_candidate=_read_bool(
                "MS_PESO_ALLOW_UNAPPROVED_CANDIDATE", default=False
            ),
            host=os.getenv("MS_PESO_HOST", "127.0.0.1").strip(),
            port=_read_int("MS_PESO_PORT", default=8080, minimum=1, maximum=65535),
        )
