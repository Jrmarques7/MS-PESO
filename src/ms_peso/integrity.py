from __future__ import annotations

import hashlib
from pathlib import Path


def calculate_sha256(path: str | Path) -> str:
    """Calcula exclusivamente a identidade SHA-256 de um arquivo."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_content_sha256(content: bytes | str) -> str:
    """Calcula exclusivamente a identidade SHA-256 de conteúdo em memória."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()
