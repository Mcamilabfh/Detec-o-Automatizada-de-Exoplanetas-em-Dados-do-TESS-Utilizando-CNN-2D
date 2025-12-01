"""Compat wrapper: gera imagens PNG chamando o novo modulo projeto_toi.data.images."""
from __future__ import annotations

from pathlib import Path

from projeto_toi.config import load_config
from projeto_toi.data.images import ImageDatasetConfig, create_image_dataset, cli


def create_dataset():
    cfg = load_config()
    create_image_dataset(
        input_dir=cfg.paths.processed,
        output_dir=cfg.paths.datasets / "images",
        cfg=ImageDatasetConfig(),
        project=cfg,
    )


if __name__ == "__main__":
    cli()
