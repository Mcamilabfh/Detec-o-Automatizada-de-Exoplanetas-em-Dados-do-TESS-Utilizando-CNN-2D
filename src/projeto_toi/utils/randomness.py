"""Helpers para reprodutibilidade."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Fixa seeds do python/numpy e, se disponível, torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def build_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Retorna um gerador numpy já configurado."""
    return np.random.default_rng(seed)
