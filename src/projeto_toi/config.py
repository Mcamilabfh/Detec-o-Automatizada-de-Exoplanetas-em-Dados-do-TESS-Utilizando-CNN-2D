"""Configurações globais e caminhos padrão."""
from __future__ import annotations

from dataclasses import dataclass

from .utils.paths import ProjectPaths, default_paths


@dataclass(frozen=True)
class ProjectConfig:
    paths: ProjectPaths
    default_cutout: int = 15
    default_seed: int = 42


def load_config() -> ProjectConfig:
    """Retorna a configuração padrão com diretórios já criados."""
    paths = default_paths().ensure()
    return ProjectConfig(paths=paths)


__all__ = ["ProjectConfig", "load_config"]
