"""Download de cutouts TESScut + quicklooks em formato funcional."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from lightkurve import search_tesscut

from ..config import ProjectConfig, load_config
from ..utils.logging import info
from ..utils.paths import ensure_dirs


@dataclass
class CutoutArtifacts:
    fits_path: Path
    cube_path: Path
    lightcurve_path: Path
    figures: Dict[str, Path]
    meta: Dict


def _target_slug(target: Optional[str], ra: Optional[float], dec: Optional[float], sector: Optional[int]) -> str:
    base = ""
    if target:
        base = target.lower().replace(" ", "_").replace("tic", "tic")
    elif ra is not None and dec is not None:
        base = f"ra{ra:.3f}_dec{dec:.3f}"
    else:
        base = "target"
    if sector is not None:
        base = f"{base}_s{sector}"
    return base


def _choose_result(search_result, sector: Optional[int]):
    if len(search_result) == 0:
        raise RuntimeError("Nenhum TESScut encontrado para esse alvo/setor.")
    if sector is not None:
        matches = [r for r in search_result if getattr(r, "sector", None) == sector]
        if not matches:
            raise RuntimeError(f"Nenhum cutout para o setor {sector}. Use sem --sector para escolher automatico.")
        return matches[0]
    try:
        return sorted(search_result, key=lambda r: getattr(r, "sector", -1))[-1]
    except Exception:
        return search_result[0]


def download_tess_cutout(
    target: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    sector: Optional[int] = None,
    cutout_size: int = 15,
    output_dir: Optional[Path] = None,
    config: Optional[ProjectConfig] = None,
) -> CutoutArtifacts:
    """
    Faz download do TESScut, salva FITS/cube/curva de luz e figuras de quicklook.
    """
    cfg = config or load_config()
    slug = _target_slug(target, ra, dec, sector)
    base_dir = Path(output_dir) if output_dir else cfg.paths.raw / slug
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    ensure_dirs([data_dir, figs_dir])

    query_target = target if target else SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    info(f"Buscando TESScut para {query_target} (sector={sector or 'auto'})")
    search_result = search_tesscut(query_target) if sector is None else search_tesscut(query_target, sector=sector)
    chosen = _choose_result(search_result, sector)
    info(f"Setor escolhido: {getattr(chosen, 'sector', 'desconhecido')}")

    info(f"Baixando cutout {cutout_size}x{cutout_size} ...")
    tpf = chosen.download(cutout_size=cutout_size)
    if tpf is None:
        raise RuntimeError("Falha no download do cutout. Tente outro alvo/setor.")

    fits_path = data_dir / "tpf.fits"
    try:
        tpf.to_fits(fits_path, overwrite=True)
    except Exception:
        tpf.hdu.writeto(fits_path, overwrite=True)

    cube = np.array(tpf.flux)
    time_btjd = np.array(tpf.time.value)
    bjd = time_btjd + 2457000.0
    time_iso = Time(bjd, format="jd").iso

    cube_path = data_dir / "cube.npy"
    np.save(cube_path, cube)

    try:
        lc = tpf.to_lightcurve(aperture_mask="threshold")
    except Exception:
        central = cube[:, cube.shape[1] // 2 - 1 : cube.shape[1] // 2 + 2, cube.shape[2] // 2 - 1 : cube.shape[2] // 2 + 2]
        flux = np.nanmean(central, axis=(1, 2))
        from lightkurve.lightcurve import TessLightCurve

        lc = TessLightCurve(time=tpf.time, flux=flux)

    lc = lc.remove_nans()
    try:
        lc_flat = lc.flatten(window_length=101)
    except Exception:
        lc_flat = lc

    lightcurve_path = data_dir / "lightcurve.csv"
    df = pd.DataFrame({"time_btjd": lc_flat.time.value, "flux": np.asarray(lc_flat.flux)})
    if hasattr(lc_flat, "flux_err") and lc_flat.flux_err is not None:
        df["flux_err"] = np.asarray(lc_flat.flux_err)
    df.to_csv(lightcurve_path, index=False)

    median_img = np.nanmedian(cube, axis=0)
    plt.figure()
    plt.imshow(median_img, origin="lower")
    plt.colorbar()
    plt.title("Imagem mediana do cutout")
    fig_image = figs_dir / "quicklook_image.png"
    plt.savefig(fig_image, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["time_btjd"], df["flux"], linewidth=0.8)
    plt.xlabel("Tempo (BTJD)")
    plt.ylabel("Fluxo (unid. relativa)")
    plt.title("Curva de luz (flatten)")
    plt.grid(True, alpha=0.3)
    fig_lc = figs_dir / "quicklook_lc.png"
    plt.savefig(fig_lc, dpi=150, bbox_inches="tight")
    plt.close()

    try:
        cx = np.array(tpf.centroid_col)
        cy = np.array(tpf.centroid_row)
    except Exception:
        y, x = np.indices(cube.shape[1:])
        flux = cube - np.nanmedian(cube, axis=0)
        flux[np.isnan(flux)] = 0.0
        total = np.sum(np.abs(flux), axis=(1, 2)) + 1e-9
        cx = np.sum(np.abs(flux) * x, axis=(1, 2)) / total
        cy = np.sum(np.abs(flux) * y, axis=(1, 2)) / total

    plt.figure()
    plt.plot(cx, cy, linewidth=0.8)
    plt.scatter(cx[:1], cy[:1])
    plt.xlabel("centroid_x (px)")
    plt.ylabel("centroid_y (px)")
    plt.title("Trajetoria do centroide ao longo do tempo")
    plt.grid(True, alpha=0.3)
    fig_centroid = figs_dir / "quicklook_centroid.png"
    plt.savefig(fig_centroid, dpi=150, bbox_inches="tight")
    plt.close()

    meta = {
        "target": target,
        "ra": ra,
        "dec": dec,
        "sector": getattr(chosen, "sector", None),
        "cutout_size": cutout_size,
        "n_frames": int(len(time_btjd)),
        "shape": list(cube.shape),
    }
    info(f"Quicklooks salvos em {figs_dir}")
    return CutoutArtifacts(
        fits_path=fits_path,
        cube_path=cube_path,
        lightcurve_path=lightcurve_path,
        figures={"image": fig_image, "lc": fig_lc, "centroid": fig_centroid},
        meta=meta,
    )


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Baixa um TESScut e gera quicklooks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=str, help='Ex.: "TIC 307210830" ou "HD 219134"')
    group.add_argument("--ra", type=float, help="Ascensao reta em graus (use com --dec)")
    parser.add_argument("--dec", type=float, help="Declinacao em graus (exige --ra)")
    parser.add_argument("--sector", type=int, default=None, help="Numero do setor TESS (opcional)")
    parser.add_argument("--cutout", type=int, default=15, help="Tamanho do recorte quadrado em pixels (default=15)")
    parser.add_argument("--outdir", type=str, default="", help="Pasta base de saida (default=data/raw/<alvo>)")
    ns = parser.parse_args(args=args)

    if ns.ra is not None and ns.dec is None:
        parser.error("--dec e obrigatorio quando --ra e fornecido.")
    config = load_config()
    outdir = Path(ns.outdir) if ns.outdir else None
    download_tess_cutout(
        target=ns.target,
        ra=ns.ra,
        dec=ns.dec,
        sector=ns.sector,
        cutout_size=ns.cutout,
        output_dir=outdir,
        config=config,
    )


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
