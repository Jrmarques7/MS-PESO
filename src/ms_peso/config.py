from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Carrega exclusivamente uma configuração YAML do projeto."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuração YAML inválida: {config_path}")
    return config
