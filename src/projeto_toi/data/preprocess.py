"""Pre-processamento: junta TPFs e gera curva dobrada pronta para gerar janelas/imagens."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ..config import ProjectConfig, load_config
from ..utils.logging import info, warn
from ..utils.paths import ensure_dirs


@dataclass
class FoldedLightcurve:
    phase: np.ndarray
    flux: np.ndarray
    tic_id: str
    path: Path
    figure: Path | None = None


def _tic_number(tic_id: str) -> str:
    return tic_id.replace("TIC", "").replace("tic", "").replace(" ", "")


def find_tpf_files(tic_id: str, search_root: Path) -> List[Path]:
    tic_num = _tic_number(tic_id)
    patterns = [f"*{tic_num}*.fits", "tpf.fits"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(search_root.rglob(pat))
    # remove duplicatas preservando ordem
    uniq = []
    seen = set()
    for p in files:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _extract_flux_time(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        flux_key = None
        for key in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX"):
            if key in data.names:
                flux_key = key
                break
        if flux_key is None:
            raise KeyError(f"Nenhum campo de fluxo padrao encontrado em {file_path.name}")
        flux_data = data[flux_key]
        if flux_key == "FLUX":
            flux = np.nansum(flux_data, axis=(1, 2) if flux_data.ndim == 3 else 1)
        else:
            flux = flux_data
        time = data["TIME"]
        valid = np.isfinite(flux) & np.isfinite(time)
        return time[valid], flux[valid]


def fold_lightcurve(
    tic_id: str,
    period_days: float,
    search_root: Path,
    output_dir: Path,
    window_length: int = 201,
    polyorder: int = 3,
    save_plot: bool = True,
    config: ProjectConfig | None = None,
) -> FoldedLightcurve:
    cfg = config or load_config()
    files = find_tpf_files(tic_id, search_root)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .fits encontrado em {search_root} para {tic_id}")

    info(f"Encontrados {len(files)} TPF(s) para {tic_id}")
    time_all: List[np.ndarray] = []
    flux_all: List[np.ndarray] = []
    for fp in files:
        try:
            t, f = _extract_flux_time(fp)
            time_all.append(t)
            flux_all.append(f)
        except Exception as exc:
            warn(f"Falha ao ler {fp.name}: {exc}")
    if not time_all:
        raise RuntimeError("Nenhum arquivo TPF valido para compor a curva.")

    time = np.concatenate(time_all)
    flux = np.concatenate(flux_all)
    lc = lk.LightCurve(time=time, flux=flux).normalize()
    flat = lc.flatten(window_length=window_length, polyorder=polyorder)
    folded = flat.fold(period=period_days)

    ensure_dirs([output_dir])
    out_path = output_dir / f"{_tic_number(tic_id)}_folded.npz"
    np.savez(out_path, phase=folded.time.value, flux=folded.flux.value, tic_id=tic_id)

    fig_path = None
    if save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        ax1.set_title(f"Curva de luz limpa ({tic_id})")
        flat.scatter(ax=ax1, s=3, color="blue")
        ax2.set_title("Transito dobrado (folded)")
        folded.errorbar(ax=ax2, marker=".", linestyle="none", color="black", alpha=0.5)
        folded.bin(bins=50).plot(ax=ax2, color="red", linewidth=3)
        ax2.set_ylabel("Fluxo normalizado")
        ax2.set_xlabel("Fase (periodo)")
        plt.tight_layout()
        fig_path = output_dir / f"{_tic_number(tic_id)}_folded.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

    return FoldedLightcurve(
        phase=folded.time.value,
        flux=folded.flux.value,
        tic_id=tic_id,
        path=out_path,
        figure=fig_path,
    )


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepara curva dobrada a partir de TPFs baixados.")
    parser.add_argument("--tic", required=True, help='ID do alvo, ex.: "TIC 52368076"')
    parser.add_argument("--period", required=True, type=float, help="Periodo orbital em dias")
    parser.add_argument("--search-root", default="", help="Raiz onde estao os FITS (default=data/raw)")
    parser.add_argument("--output-dir", default="", help="Diretorio de saida (default=data/processed)")
    parser.add_argument("--no-plot", action="store_true", help="Nao salvar figura de diagnostico")
    ns = parser.parse_args(args=args)

    cfg = load_config()
    search_root = Path(ns.search_root) if ns.search_root else cfg.paths.raw
    output_dir = Path(ns.output_dir) if ns.output_dir else cfg.paths.processed
    fold_lightcurve(
        tic_id=ns.tic,
        period_days=ns.period,
        search_root=search_root,
        output_dir=output_dir,
        save_plot=not ns.no_plot,
        config=cfg,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
