"""Compat wrapper: pre-processa curvas chamando projeto_toi.data.preprocess."""
from __future__ import annotations

from projeto_toi.data.preprocess import cli, fold_lightcurve

if __name__ == "__main__":
    cli()
