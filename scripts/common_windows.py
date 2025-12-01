"""Compat layer re-exportando helpers de janelas do pacote principal."""
from __future__ import annotations

from projeto_toi.data.windows import *  # noqa: F401,F403
from projeto_toi.utils.randomness import build_rng, seed_everything  # reexports
