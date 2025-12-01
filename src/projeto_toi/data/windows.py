"""Criacao de janelas POS/NEG a partir de TPFs ja baixados."""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from ..config import ProjectConfig, load_config
from ..utils.logging import info
from ..utils.randomness import build_rng, seed_everything


@dataclass
class WindowConfig:
    pos_margin_durations: float = 1.5
    neg_per_pos: int = 2
    min_gap_factor: float = 2.0
    seed: int = 42


@dataclass
class WindowArtifacts:
    pos_paths: List[Path]
    neg_paths: List[Path]
    meta: Dict


# ------------------------------------------------------------------ #
# TPF helpers
# ------------------------------------------------------------------ #
def find_tpf_path(outdir: Path) -> Path:
    """Procura um tpf.fits dentro de outdir."""
    cands = [outdir / "data" / "tpf.fits", outdir / "tpf.fits"]
    for c in cands:
        if c.exists():
            return c
    for p in outdir.rglob("tpf.fits"):
        return p
    raise FileNotFoundError(f"Nao achei tpf.fits em {outdir}")


def load_tpf(outdir: Path):
    from lightkurve import TessTargetPixelFile

    tpf_path = find_tpf_path(outdir)
    return TessTargetPixelFile(tpf_path)


def extract_tic_sector(tpf) -> Tuple[Optional[str], Optional[str]]:
    """Extrai TIC e setor se presentes no TPF."""
    tic = getattr(tpf, "targetid", None)
    sector = getattr(tpf, "sector", None)
    tic_s = str(tic) if tic is not None else None
    sector_s = str(sector) if sector is not None else None
    return tic_s, sector_s


# ------------------------------------------------------------------ #
# Janelas
# ------------------------------------------------------------------ #
def centers_in_btjd(time_btjd: np.ndarray, period_days: float, epoch_bjd: float) -> List[float]:
    epoch_btjd = epoch_bjd - 2457000.0
    tmin, tmax = float(np.nanmin(time_btjd)), float(np.nanmax(time_btjd))
    k_start = int(np.floor((tmin - epoch_btjd) / period_days) - 1)
    k_end = int(np.ceil((tmax - epoch_btjd) / period_days) + 1)
    centers = []
    for k in range(k_start, k_end + 1):
        tc = epoch_btjd + k * period_days
        if tmin <= tc <= tmax:
            centers.append(tc)
    return centers


def slice_windows(
    cube: np.ndarray,
    t: np.ndarray,
    centers: Sequence[float],
    duration_hours: float,
    pos_margin: float = 1.5,
    neg_per_pos: int = 2,
    min_gap_factor: float = 2.0,
    rng: Optional[np.random.Generator] = None,
):
    """Separa janelas POS/NEG a partir de centros de transito."""
    rng = rng or np.random.default_rng()
    dur_days = duration_hours / 24.0
    half_win = 0.5 * dur_days * pos_margin
    pos, neg = [], []

    for tc in centers:
        idx = np.where((t >= tc - half_win) & (t <= tc + half_win))[0]
        if len(idx) >= 5:
            pos.append({"idx": idx, "label": 1, "center_btjd": tc})

    min_gap = 0.5 * dur_days * min_gap_factor
    forbidden = np.zeros_like(t, dtype=bool)
    for tc in centers:
        forbidden |= (np.abs(t - tc) <= min_gap)
    eligible = np.where(~forbidden)[0]

    for _ in pos:
        added = 0
        tries = 0
        while added < neg_per_pos and tries < 200 and len(eligible) > 0:
            tries += 1
            j = int(rng.choice(eligible))
            tc_neg = t[j]
            idx = np.where((t >= tc_neg - half_win) & (t <= tc_neg + half_win))[0]
            if len(idx) >= 5 and not np.any(forbidden[idx]):
                neg.append({"idx": idx, "label": 0, "center_btjd": tc_neg})
                added += 1
    return pos, neg


def save_windows(
    outdir: Path,
    cube: np.ndarray,
    t: np.ndarray,
    wins: List[dict],
    prefix: str,
    subdir: str = "windows",
    extra_meta: Optional[Dict] = None,
    duration_hours: Optional[float] = None,
) -> List[Path]:
    wdir = outdir / "data" / subdir
    wdir.mkdir(parents=True, exist_ok=True)
    extra_meta = extra_meta or {}
    saved: List[Path] = []
    for k, w in enumerate(wins):
        subcube = cube[w["idx"], :, :]
        subtime = t[w["idx"]]
        mask_during = None
        mask_before = None
        idx_in_transit = None
        if duration_hours is not None:
            half_dur_days = 0.5 * float(duration_hours) / 24.0
            mask_during = np.abs(subtime - float(w["center_btjd"])) <= half_dur_days
            if mask_during.sum() >= 1:
                mask_before = ~mask_during
                idx_in_transit = np.where(mask_during)[0].astype(np.int64)
            else:
                mask_during = None
        meta = {
            "label": int(w["label"]),
            "center_btjd": float(w["center_btjd"]),
            "n_frames": int(len(subtime)),
            "shape": list(subcube.shape),
        }
        meta.update(extra_meta)
        path = wdir / f"{prefix}_{k:03d}.npz"
        np.savez_compressed(
            path,
            cube=subcube,
            time_btjd=subtime,
            meta=json.dumps(meta),
            idx_in_transit=idx_in_transit,
            mask_before=mask_before,
            mask_during=mask_during,
        )
        saved.append(path)
    return saved


# ------------------------------------------------------------------ #
# TOI
# ------------------------------------------------------------------ #
def fetch_toi_params_by_tic(tic_id: int) -> Optional[Dict[str, float]]:
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+*+from+toi+where+tic_id%3D{tic}&format=json"
    ).format(tic=int(tic_id))
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        row = data[0]

        def pick(keys):
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        period_days = to_float(pick(["Period", "pl_orbper", "period", "pl_orbper_d"]))
        epoch_bjd = to_float(pick(["Epoch", "pl_tranmid", "epoch", "pl_tranmid_bjd"]))
        dur_hours = to_float(pick(["Duration", "pl_trandurh", "duration", "pl_trandurh_hours"]))
        if None in (period_days, epoch_bjd, dur_hours):
            return None
        return {"period_days": period_days, "epoch_bjd": epoch_bjd, "duration_hours": dur_hours}
    except Exception as exc:
        warnings.warn(f"Falha no fetch TOI: {exc}")
        return None


def make_windows_from_tpf(
    outdir: Path,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    cfg: WindowConfig,
    project: ProjectConfig | None = None,
) -> WindowArtifacts:
    project = project or load_config()
    seed_everything(cfg.seed)
    rng = build_rng(cfg.seed)

    tpf = load_tpf(outdir)
    cube = np.array(tpf.flux)
    time_btjd = np.array(tpf.time.value)
    tic, sector = extract_tic_sector(tpf)

    centers = centers_in_btjd(time_btjd, period_days, epoch_bjd)
    if not centers:
        raise SystemExit("Nenhum centro de transito caiu na janela observada.")
    pos, neg = slice_windows(
        cube,
        time_btjd,
        centers,
        duration_hours,
        pos_margin=cfg.pos_margin_durations,
        neg_per_pos=cfg.neg_per_pos,
        min_gap_factor=cfg.min_gap_factor,
        rng=rng,
    )

    meta = {
        "tic": tic,
        "sector": sector,
        "period_days": period_days,
        "epoch_bjd": epoch_bjd,
        "duration_hours": duration_hours,
        "seed": cfg.seed,
    }
    sp = save_windows(outdir, cube, time_btjd, pos, "pos", subdir="windows", extra_meta=meta, duration_hours=duration_hours)
    sn = save_windows(outdir, cube, time_btjd, neg, "neg", subdir="windows", extra_meta=meta, duration_hours=duration_hours)
    info(f"OK: POS={len(sp)} NEG={len(sn)} -> {outdir / 'data' / 'windows'}")
    return WindowArtifacts(pos_paths=sp, neg_paths=sn, meta=meta)


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gera janelas POS/NEG a partir do tpf.fits")
    parser.add_argument("--outdir", default="", help="Pasta onde o tpf.fits esta salvo (default=data/raw/<alvo>)")
    parser.add_argument("--tic", type=int, help="TIC opcional para buscar parametros TOI")
    parser.add_argument("--period-days", type=float, help="Periodo em dias (obrigatorio se nao usar --tic)")
    parser.add_argument("--epoch-bjd", type=float, help="Epoca em BJD (obrigatorio se nao usar --tic)")
    parser.add_argument("--duration-hours", type=float, help="Duracao do transito em horas (obrigatorio se nao usar --tic)")
    parser.add_argument("--pos-margin-durations", type=float, default=1.5)
    parser.add_argument("--neg-per-pos", type=int, default=2)
    parser.add_argument("--min-gap-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args(args=args)

    cfg = load_config()
    outdir = Path(ns.outdir) if ns.outdir else cfg.paths.raw
    params = None
    if ns.tic:
        params = fetch_toi_params_by_tic(ns.tic)
    if ns.period_days is not None:
        params = params or {}
        params["period_days"] = ns.period_days
    if ns.epoch_bjd is not None:
        params = params or {}
        params["epoch_bjd"] = ns.epoch_bjd
    if ns.duration_hours is not None:
        params = params or {}
        params["duration_hours"] = ns.duration_hours
    if not params or any(params.get(k) is None for k in ("period_days", "epoch_bjd", "duration_hours")):
        raise SystemExit("Faltam parametros. Use --tic OU passe --period-days --epoch-bjd --duration-hours.")

    wcfg = WindowConfig(
        pos_margin_durations=ns.pos_margin_durations,
        neg_per_pos=ns.neg_per_pos,
        min_gap_factor=ns.min_gap_factor,
        seed=ns.seed,
    )
    make_windows_from_tpf(
        outdir=outdir,
        period_days=params["period_days"],
        epoch_bjd=params["epoch_bjd"],
        duration_hours=params["duration_hours"],
        cfg=wcfg,
        project=cfg,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
