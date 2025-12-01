"""
Gerenciamento simples de caminhos do projeto.

Os caminhos podem ser sobrescritos por variáveis de ambiente:
- PROJETO_TOI_ROOT
- PROJETO_TOI_DATA
- PROJETO_TOI_ARTIFACTS
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProjectPaths:
    """Coleção de caminhos canônicos usados no pipeline."""

    root: Path
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    datasets: Path
    artifacts: Path
    runs: Path
    models: Path
    figures: Path

    def ensure(self) -> "ProjectPaths":
        """Garante que os diretórios principais existam."""
        for p in (self.data_root, self.raw, self.interim, self.processed, self.datasets, self.artifacts, self.runs, self.models, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _env_path(var_name: str, default: Path) -> Path:
    return Path(os.environ.get(var_name, default))


def default_paths(root: Path | None = None) -> ProjectPaths:
    """Constroi os caminhos com base na raiz do repo (ou variável de ambiente)."""
    repo_root = Path(_env_path("PROJETO_TOI_ROOT", root or Path(__file__).resolve().parents[2])).resolve()
    data_root = _env_path("PROJETO_TOI_DATA", repo_root / "data").resolve()
    artifacts = _env_path("PROJETO_TOI_ARTIFACTS", repo_root / "artifacts").resolve()
    return ProjectPaths(
        root=repo_root,
        data_root=data_root,
        raw=data_root / "raw",
        interim=data_root / "interim",
        processed=data_root / "processed",
        datasets=data_root / "datasets",
        artifacts=artifacts,
        runs=artifacts / "runs",
        models=artifacts / "models",
        figures=artifacts / "figures",
    )


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Cria diretórios informados, se necessário."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
