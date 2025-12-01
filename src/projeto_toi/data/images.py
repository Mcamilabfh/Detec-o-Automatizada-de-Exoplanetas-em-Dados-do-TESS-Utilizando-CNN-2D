"""Converte janelas de fase em PNGs para uso em visao computacional."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from ..config import ProjectConfig, load_config
from ..utils.logging import info
from ..utils.paths import ensure_dirs


@dataclass
class ImageDatasetConfig:
    transit_fraction: float = 0.1  # fracao do periodo usada como janela


def save_window_as_image(phase_window: np.ndarray, flux_window: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(phase_window, flux_window, s=5, color="black", alpha=0.7)
    ax.axis("off")
    ax.set_ylim(min(flux_window) * 0.999, max(flux_window) * 1.001)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


def _windows_from_npz(npz_path: Path, cfg: ImageDatasetConfig, output_dir: Path) -> List[Path]:
    data = np.load(npz_path, allow_pickle=True)
    phase = data["phase"]
    flux = data["flux"]
    tic_id = str(data["tic_id"].item()) if "tic_id" in data else "tic"
    frac = cfg.transit_fraction
    saved: List[Path] = []

    pos_mask = (phase >= -frac / 2) & (phase <= frac / 2)
    if np.sum(pos_mask) > 5:
        out = output_dir / "positive" / f"{tic_id}_positive_001.png"
        save_window_as_image(phase[pos_mask], flux[pos_mask], out)
        saved.append(out)

    neg1_center = 0.25
    neg1_mask = (phase >= neg1_center - frac / 2) & (phase <= neg1_center + frac / 2)
    if np.sum(neg1_mask) > 5:
        out = output_dir / "negative" / f"{tic_id}_negative_001.png"
        save_window_as_image(phase[neg1_mask], flux[neg1_mask], out)
        saved.append(out)

    neg2_center = 0.75
    neg2_mask = (phase >= neg2_center - frac / 2) & (phase <= neg2_center + frac / 2)
    if np.sum(neg2_mask) > 5:
        out = output_dir / "negative" / f"{tic_id}_negative_002.png"
        save_window_as_image(phase[neg2_mask], flux[neg2_mask], out)
        saved.append(out)
    return saved


def create_image_dataset(
    input_dir: Path,
    output_dir: Path,
    cfg: ImageDatasetConfig | None = None,
    project: ProjectConfig | None = None,
) -> List[Path]:
    cfg = cfg or ImageDatasetConfig()
    project = project or load_config()
    ensure_dirs([output_dir])
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"Nenhum arquivo .npz encontrado em {input_dir}")
    saved: List[Path] = []
    for f in npz_files:
        saved.extend(_windows_from_npz(f, cfg, output_dir))
    info(f"{len(saved)} imagens salvas em {output_dir}")
    return saved


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cria imagens PNG a partir de janelas dobradas.")
    parser.add_argument("--input", default="", help="Diretorio com arquivos .npz dobrados (default=data/processed)")
    parser.add_argument("--output", default="", help="Diretorio de saida das imagens (default=data/datasets/images)")
    parser.add_argument("--transit-fraction", type=float, default=0.1, help="Fracao do periodo usada na janela")
    ns = parser.parse_args(args=args)

    cfg = load_config()
    input_dir = Path(ns.input) if ns.input else cfg.paths.processed
    output_dir = Path(ns.output) if ns.output else cfg.paths.datasets / "images"
    create_image_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        cfg=ImageDatasetConfig(transit_fraction=ns.transit_fraction),
        project=cfg,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
