#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gerador/rotulador automatico de janelas (sem TOI) usando BLS (Astropy) + Lightkurve.

Uso (exemplos):
  # No diretorio de um alvo que contenha data/tpf.fits
  python scripts/make_windows_auto_bls.py --outdir . --max_period 30 --neg_per_pos 2

  # Em uma pasta especifica (que contenha data/tpf.fits OU tpf.fits)
  python scripts/make_windows_auto_bls.py --outdir data/TIC_307210830/run_20251007_1449 \
      --max_period 30 --neg_per_pos 2 --min_bls_power 0.5
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

# Ensure repo root is importable when executing from scripts/
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from lightkurve.lightcurve import TessLightCurve
from astropy.timeseries import BoxLeastSquares

from scripts.common_windows import (
    build_rng,
    centers_in_btjd,
    extract_tic_sector,
    load_tpf,
    save_windows,
    seed_everything,
    slice_windows,
)


# ---------------------------------------------------------------------
# Preparacao da curva
# ---------------------------------------------------------------------
def get_clean_flat_lc(tpf) -> TessLightCurve:
    """Extrai uma light curve do TPF e aplica flatten; robusto a falhas no aperture mask."""
    try:
        lc = tpf.to_lightcurve(aperture_mask="threshold")
    except Exception:
        cube = np.array(tpf.flux)
        # janela central 2x2 como fallback
        h, w = cube.shape[1], cube.shape[2]
        central = cube[:, h // 2 - 1: h // 2 + 1, w // 2 - 1: w // 2 + 1]
        flux = np.nanmean(central, axis=(1, 2))
        lc = TessLightCurve(time=tpf.time, flux=flux)

    lc = lc.remove_nans()
    try:
        lc = lc.remove_outliers(sigma=5.0)
    except Exception:
        pass
    try:
        lc_flat = lc.flatten(window_length=101)
    except Exception:
        lc_flat = lc
    return lc_flat


# ---------------------------------------------------------------------
# BLS (Astropy) com quedas para diferentes versoes
# ---------------------------------------------------------------------
def compute_bls_astropy(
    lc_flat: TessLightCurve,
    min_period: float,
    max_period: float,
    min_duration_hours: float = 0.5,
    max_duration_hours: float = 8.0,
    samples_per_peak: int = 5,
    n_durations: int = 15,
):
    """Executa BoxLeastSquares.autopower de forma compativel com versoes antigas/novas."""
    t = np.asarray(lc_flat.time.value)  # BTJD
    y = np.asarray(lc_flat.flux)
    try:
        dy = np.asarray(lc_flat.flux_err)
        if dy is None or np.all(np.isnan(dy)):
            dy = None
    except Exception:
        dy = None

    bls = BoxLeastSquares(t, y, dy) if dy is not None else BoxLeastSquares(t, y)
    durations = np.linspace(min_duration_hours / 24.0, max_duration_hours / 24.0, n_durations)

    res = None
    used = ""

    # Tentativa 1: versoes que aceitam minimum/maximum_period + samples_per_peak
    try:
        res = bls.autopower(
            durations,
            minimum_period=min_period,
            maximum_period=max_period,
            samples_per_peak=samples_per_peak,
        )
        used = "autopower(min/max_period + samples_per_peak)"
    except TypeError:
        # Tentativa 2: min/max sem samples_per_peak
        try:
            res = bls.autopower(
                durations,
                minimum_period=min_period,
                maximum_period=max_period,
            )
            used = "autopower(min/max_period)"
        except TypeError:
            # Tentativa 3: grid de frequencia
            nfreq = 5000
            freq = np.linspace(1.0 / max_period, 1.0 / min_period, nfreq)
            res = bls.autopower(durations, frequency=freq)
            used = "autopower(frequency grid)"

    k = int(np.nanargmax(res.power))
    P = float(res.period[k])
    T0 = float(res.transit_time[k])
    D = float(res.duration[k])
    # retornamos tambem os arrays para calculo de pmax no main()
    return P, T0, D, used, (res.period, res.power, res.duration, res.transit_time)


# ---------------------------------------------------------------------
# Meta do TPF
# ---------------------------------------------------------------------
def safe_meta_from_tpf(tpf) -> Tuple[str, str]:
    """Tenta extrair TIC e SECTOR do TPF (se disponivel)."""
    return extract_tic_sector(tpf)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="Pasta-base do alvo (contendo 'data/tpf.fits' ou 'tpf.fits').")

    # Busca do periodo
    ap.add_argument("--min_period", type=float, default=0.5, help="Periodo minimo (dias).")
    ap.add_argument("--max_period", type=float, default=10.0, help="Periodo maximo (dias).")
    ap.add_argument("--min_duration_hours", type=float, default=0.5)
    ap.add_argument("--max_duration_hours", type=float, default=8.0)
    ap.add_argument("--samples_per_peak", type=int, default=5)
    ap.add_argument("--n_durations", type=int, default=15)

    # Construcao das janelas
    ap.add_argument("--neg_per_pos", type=int, default=2, help="Razao de negativos por positivo.")
    ap.add_argument(
        "--pos_margin_durations",
        type=float,
        default=1.5,
        help="Largura da janela POS em multiplos da duracao (default=1.5).",
    )
    ap.add_argument(
        "--min_gap_factor",
        type=float,
        default=2.0,
        help="Minima distancia (em multiplos de 0.5*dur) para NEG ficar longe dos centros (default=2.0).",
    )

    # Novo: filtro para potencia minima do BLS
    ap.add_argument(
        "--min_bls_power",
        type=float,
        default=0.5,
        help="Limiar minimo da potencia do pico BLS para aceitar o alvo (default=0.5).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade.")

    args = ap.parse_args()

    seed_everything(args.seed)
    rng = build_rng(args.seed)

    # Carrega TPF
    tpf = load_tpf(args.outdir)
    cube = np.array(tpf.flux)  # (T,H,W)
    t = np.array(tpf.time.value)  # BTJD

    # Meta (opcional)
    tic, sector = safe_meta_from_tpf(tpf)

    # Curva limpa e BLS
    lc_flat = get_clean_flat_lc(tpf)
    P, T0, D, used, bls_arrays = compute_bls_astropy(
        lc_flat,
        args.min_period,
        args.max_period,
        args.min_duration_hours,
        args.max_duration_hours,
        args.samples_per_peak,
        args.n_durations,
    )

    # pmax e filtro
    _, power_arr, _, _ = bls_arrays
    pmax = float(np.nanmax(power_arr))

    print(f"[BLS] periodo={P:.5f} d  T0(BTJD)={T0:.5f}  duracao={D*24:.2f} h  | modo={used}  | pmax={pmax:.3f}")
    if tic or sector:
        print(f"[META] TIC={tic}  SECTOR={sector}")

    if pmax < args.min_bls_power:
        print(f"[SKIP] pico BLS fraco (power={pmax:.3f} < {args.min_bls_power}) - nenhuma janela sera salva.")
        raise SystemExit(2)

    # Janelas POS/NEG
    centers = centers_in_btjd(t, P, T0 + 2457000.0)
    pos, neg = slice_windows(
        cube,
        t,
        centers,
        duration_hours=D * 24.0,
        pos_margin=args.pos_margin_durations,
        neg_per_pos=args.neg_per_pos,
        min_gap_factor=args.min_gap_factor,
        rng=rng,
    )

    meta = {
        "tic": tic,
        "sector": sector,
        "period_days": P,
        "epoch_bjd": T0 + 2457000.0,
        "duration_hours": D * 24.0,
        "source": "bls",
        "seed": args.seed,
        "bls_power": pmax,
    }
    sp = save_windows(args.outdir, cube, t, pos, "pos", subdir="windows_auto", extra_meta=meta)
    sn = save_windows(args.outdir, cube, t, neg, "neg", subdir="windows_auto", extra_meta=meta)
    print(f"[DONE] windows_auto/: POS={len(sp)} NEG={len(sn)}")

    # Curva em fase (apenas visual)
    try:
        phase = ((lc_flat.time.value - T0 + 0.5 * P) % P) - 0.5 * P
        figsdir = os.path.join(args.outdir, "figs")
        os.makedirs(figsdir, exist_ok=True)
        plt.figure()
        plt.scatter(phase, lc_flat.flux, s=3, alpha=0.5)
        plt.xlabel("Fase (dias)")
        plt.ylabel("Fluxo (flatten)")
        plt.title("Curva em fase (BLS - Astropy)")
        plt.savefig(os.path.join(figsdir, "bls_phase.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("[OK] figs/bls_phase.png salvo.")
    except Exception as e:
        print(f"[WARN] plot fase nao gerado: {e}")


if __name__ == "__main__":
    main()
