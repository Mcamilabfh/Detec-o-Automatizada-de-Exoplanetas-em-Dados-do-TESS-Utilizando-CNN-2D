"""Dataset baseado em arquivos .npz com cubos TESS."""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_SIZE = 15  # tamanho alvo para padronizar janelas


# ---------- Helpers de carregamento ----------
def _safe_item(meta):
    try:
        return meta.item() if hasattr(meta, "item") else meta
    except Exception:
        return meta


def load_npz(npz_path: Path):
    return np.load(npz_path, allow_pickle=True)


# ---------- Formacao de canais ----------
def pad_to_square(x: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Padding central para atingir tamanho quadrado 'size'."""
    c, h, w = x.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return x
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    return np.pad(x, ((0, 0), (pt, pb), (pl, pr)), mode="edge")


def make_channels_from_cube(cube: np.ndarray) -> np.ndarray:
    # cube: (T,H,W)
    median = np.nanmedian(cube, axis=0)
    std = np.nanstd(cube, axis=0) + 1e-8
    minv = np.nanmin(cube, axis=0)
    median_minus_min = median - minv
    X = np.stack([median, std, minv, median_minus_min], axis=0)  # (C,H,W)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return pad_to_square(X)


def make_channels_from_masks(z) -> np.ndarray | None:
    """Se houver mascaras before/during ou idx_in_transit, usa canais [before, during, diff, std_total]."""
    cube = z["cube"]  # (T,H,W)
    T, H, W = cube.shape
    mask_before = z.get("mask_before")
    mask_during = z.get("mask_during")
    idx_in_transit = z.get("idx_in_transit")

    if mask_before is not None and len(mask_before) != T:
        mask_before = None
    if mask_during is not None and len(mask_during) != T:
        mask_during = None

    if mask_before is not None and mask_during is not None:
        mb = mask_before.astype(bool)
        md = mask_during.astype(bool)
        if mb.sum() < 1 or md.sum() < 1:
            return None
        before = np.nanmean(cube[mb], axis=0)
        during = np.nanmean(cube[md], axis=0)
    elif idx_in_transit is not None and len(idx_in_transit) > 0:
        it = np.array(idx_in_transit, dtype=int)
        it = it[(it >= 0) & (it < T)]
        if it.size == 0:
            return None
        during = np.nanmean(cube[it], axis=0)
        not_it = np.setdiff1d(np.arange(T), it)
        if not_it.size == 0:
            return None
        k = max(1, int(0.2 * not_it.size))
        before = np.nanmean(cube[not_it[:k]], axis=0)
    else:
        return None

    std_total = np.nanstd(cube, axis=0) + 1e-8
    diff = during - before
    X = np.stack([before, during, diff, std_total], axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return pad_to_square(X)


# ---------- Normalizacao e augment ----------
def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    x = x.copy()
    for i in range(c):
        mu = x[i].mean()
        sd = x[i].std() + 1e-8
        x[i] = (x[i] - mu) / sd
    return x


def random_augment(x: np.ndarray) -> np.ndarray:
    # flips aleatorios e pequena translacao
    if random.random() < 0.5:
        x = x[:, :, ::-1]
    if random.random() < 0.5:
        x = x[:, ::-1, :]
    pad = 2
    if random.random() < 0.8:
        C, H, W = x.shape
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        dy = random.randint(-pad, pad)
        dx = random.randint(-pad, pad)
        ys = pad + dy
        xs = pad + dx
        x = xp[:, ys : ys + H, xs : xs + W]
    return x


# ---------- Label e grupo ----------
def infer_label_from_path(path: Path) -> int:
    s = str(path).lower()
    if "/pos/" in s or re.search(r"(^|[_\\-])pos([_\\-\\.]|$)", path.name.lower()):
        return 1
    if "/neg/" in s or re.search(r"(^|[_\\-])neg([_\\-\\.]|$)", path.name.lower()):
        return 0
    raise ValueError(f"Nao foi possivel inferir label de {path}")


def infer_group_from_meta_or_name(p: Path, meta) -> str:
    if meta is not None:
        m = _safe_item(meta)
        try:
            tic = str(m.get("tic") or m.get("TIC") or m.get("target_id") or "")
            sec = str(m.get("sector") or m.get("Sector") or "")
            if tic and sec:
                return f"TIC{tic}_S{sec}"
            if tic:
                return f"TIC{tic}"
        except Exception:
            pass
    name = p.stem.lower()
    mt = re.search(r"tic[_\\-]?(\\d+)", name)
    ms = re.search(r"s(ec(tor)?)?[_\\-]?(\\d+)", name)
    if mt and ms:
        return f"TIC{mt.group(1)}_S{ms.group(4)}"
    if mt:
        return f"TIC{mt.group(1)}"
    return p.stem


# ---------- Dataset ----------
class NPZWindowsDataset(Dataset):
    def __init__(self, root_dir: Path | str, file_list: Optional[Sequence[Path | str]] = None, train: bool = True):
        self.root = Path(root_dir)
        if file_list is None:
            self.files = sorted(self.root.rglob("*.npz"))
        else:
            self.files = [Path(f) for f in file_list]
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        p = self.files[idx]
        z = load_npz(p)
        cube = z["cube"]
        meta = z.get("meta")

        Xm = make_channels_from_masks(z)
        X = Xm if Xm is not None else make_channels_from_cube(cube)

        if self.train:
            X = random_augment(X)
        X = zscore_per_channel(X).astype(np.float32)

        y = infer_label_from_path(p)
        key = p.stem
        group = infer_group_from_meta_or_name(p, meta)

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32), key, group
