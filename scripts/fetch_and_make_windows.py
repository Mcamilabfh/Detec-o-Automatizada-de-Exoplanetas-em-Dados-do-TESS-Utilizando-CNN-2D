#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_and_make_windows.py
---------------------------------
Baixa TPFs/cutouts para uma lista de TICs e gera janelas automaticas (pos/neg)
chamando o make_windows_auto_bls.py em cada diretorio baixado. No final, pode
regenerar a lista mestre all_windows.txt.

Uso (PowerShell):
  python scripts\\fetch_and_make_windows.py `
    --outroot . `
    --tics 290131778 394137592 231702397 52368076 `
    --max_sectors 1 `
    --tesscut_size 11 `
    --max_period 30 `
    --neg_per_pos 2 `
    --regen_all_list
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import warnings
from lightkurve import search_targetpixelfile, search_tesscut, TessTargetPixelFile


# ------------------------- utilitarios FS/exec ------------------------- #

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], cwd: Optional[Path] = None) -> int:
    print(f"[EXEC] {' '.join(cmd)} (cwd={cwd or Path.cwd()})")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _print_header(title: str) -> None:
    print("\n" + "=" * 3 + f" {title} " + "=" * 3)


def _final_tpf_path(root: Path) -> Path:
    return root / "data" / "tpf.fits"


# ------------------------- helpers de materializacao ------------------------- #

def _probe_candidates(dest: Path, tic: int) -> Optional[Path]:
    """Procura arquivos .fits plausiveis quando o .download() nao retorna objeto."""
    candidates: List[Path] = []
    patterns = [
        "*.fits", "*-targ.fits", "*_targ.fits",
        f"TIC {tic}-targ.fits", f"TIC_{tic}-targ.fits",
        f"*{tic}*.fits",
    ]
    for pat in patterns:
        candidates += list(dest.glob(pat))
        candidates += list(Path.cwd().glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _materialize_return(obj, final: Path, dest: Path, tic: int) -> bool:
    """
    Garante que 'final' exista como um .fits valido.
    Aceita: caminho, TessTargetPixelFile, objeto com .filename, ou None (varre disco).
    """
    wrote = False

    if isinstance(obj, (str, os.PathLike)):
        src = Path(obj)
        ensure_dir(final.parent)
        if src.resolve() != final.resolve():
            shutil.copy2(src, final)
        wrote = True

    elif isinstance(obj, TessTargetPixelFile):
        ensure_dir(final.parent)
        try:
            obj.to_fits().writeto(final, overwrite=True)
        except Exception:
            try:
                obj.hdu.writeto(final, overwrite=True)
            except Exception:
                if hasattr(obj, "filename") and obj.filename:
                    shutil.copy2(Path(obj.filename), final)
                else:
                    wrote = False
        else:
            wrote = True

    elif hasattr(obj, "filename") and getattr(obj, "filename"):
        ensure_dir(final.parent)
        shutil.copy2(Path(getattr(obj, "filename")), final)
        wrote = True

    if not wrote:
        newest = _probe_candidates(dest, tic)
        if newest:
            ensure_dir(final.parent)
            shutil.copy2(newest, final)
            wrote = True

    return wrote


# ------------------------- download (API .download) ------------------------- #

def _download_first_searchresult(sr, where: Path, tic: int) -> Optional[Path]:
    """
    Baixa apenas o primeiro item de um SearchResult usando .download().
    Retorna caminho do tpf.fits final se der certo, senao None.
    """
    if sr is None or len(sr) == 0:
        return None

    ensure_dir(where / "data")
    final = _final_tpf_path(where)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = sr[0].download(download_dir=str(where))
    except Exception as e:
        print(f"[WARN] .download() falhou: {e}")
        obj = None

    if _materialize_return(obj, final, dest=where, tic=tic) and final.exists():
        return final
    return None


def download_tpf_or_tesscut(
    tic: int,
    outdir: Path,
    max_sectors: int = 1,
    tesscut_size: int = 11,
    overwrite: bool = True
) -> List[Path]:
    """
    Tenta TPF direto; se falhar, tenta TessCut. Retorna lista de diretorios
    com 'data/tpf.fits' pronto para uso.
    """
    ok_dirs: List[Path] = []
    tic_root = outdir / "data" / f"TIC_{tic}"
    ensure_dir(tic_root)

    _print_header(f"TIC {tic}")

    # 1) TPF direto
    try:
        sr_tpf = search_targetpixelfile(f"TIC {tic}", mission="TESS")
    except Exception as e:
        print(f"[WARN] search_targetpixelfile falhou TIC {tic}: {e}")
        sr_tpf = None

    if sr_tpf:
        sector_root = tic_root / "sector_00"
        final = _download_first_searchresult(sr_tpf, sector_root, tic)
        if final and final.exists():
            ok_dirs.append(sector_root)

    # 2) TessCut (se nada acima funcionou)
    if not ok_dirs:
        try:
            sr_cut = search_tesscut(f"TIC {tic}")
        except Exception as e:
            print(f"[WARN] search_tesscut falhou TIC {tic}: {e}")
            sr_cut = None

        if sr_cut:
            sector_root = tic_root / "sector_00"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = sr_cut[0].download(
                        cutout_size=tesscut_size,
                        download_dir=str(sector_root)
                    )
            except Exception as e:
                print(f"[WARN] TessCut .download() falhou TIC {tic}: {e}")
                obj = None

            final = _final_tpf_path(sector_root)
            if _materialize_return(obj, final, dest=sector_root, tic=tic) and final.exists():
                ok_dirs.append(sector_root)

    return ok_dirs


# ------------------------- pipeline por TIC ------------------------- #

def call_make_windows(dir_: Path, max_period: float, neg_per_pos: int) -> None:
    tpf = _final_tpf_path(dir_)
    if not tpf.exists():
        print(f"[WARN] {tpf} não encontrado; pulando geração de janelas.")
        return

    cmd = [
        "python", "scripts/make_windows_auto_bls.py",
        "--outdir", str(dir_),
        "--max_period", str(max_period),
        "--neg_per_pos", str(neg_per_pos),
    ]
    rc = _run(cmd)
    if rc == 0:
        print("[OK] Janelas geradas em", dir_ / "data" / "windows_auto")
    else:
        print("[WARN] make_windows_auto_bls.py retornou código", rc, "- descartando este alvo.")


def regenerate_all_list(outroot: Path) -> Path:
    lists_dir = outroot / "runs" / "exp_windows_auto" / "lists"
    ensure_dir(lists_dir)
    out_list = lists_dir / "all_windows.txt"

    print("\n[SCAN] varrendo por janelas *.npz...")
    npz_files: List[Path] = []
    for p in outroot.rglob("windows_auto"):
        if p.is_dir():
            npz_files.extend(sorted(p.glob("*.npz")))

    with out_list.open("w", encoding="utf-8") as f:
        for p in npz_files:
            f.write(str(p.resolve()) + "\n")

    print(f"[OK] Lista mestre regenerada: {out_list}  (total {len(npz_files)} janelas)")
    return out_list


def process_tic(
    tic: int,
    outroot: Path,
    max_sectors: int,
    tesscut_size: int,
    max_period: float,
    neg_per_pos: int
) -> Tuple[int, int]:
    ok = 0
    fail = 0
    try:
        dirs = download_tpf_or_tesscut(
            tic=tic,
            outdir=outroot,
            max_sectors=max_sectors,
            tesscut_size=tesscut_size,
            overwrite=True
        )
        if not dirs:
            print(f"[WARN] Sem TPF/cutout utilizavel para TIC {tic}.")
            return (0, 1)

        for d in dirs:
            call_make_windows(d, max_period=max_period, neg_per_pos=neg_per_pos)
            ok += 1
    except Exception as e:
        print(f"[WARN] Falhou processamento do TIC {tic}: {e}")
        fail += 1
    return (ok, fail)


# ------------------------- CLI ------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Baixa TPF/TessCut e gera janelas automaticas.")
    ap.add_argument("--outroot", type=str, default=".", help="Raiz de saida (ex.: '.')")
    ap.add_argument("--tics", type=int, nargs="+", required=True, help="Lista de TICs")
    ap.add_argument("--max_sectors", type=int, default=1, help="Max. de setores por TIC (mantido por compat.)")
    ap.add_argument("--tesscut_size", type=int, default=11, help="Tamanho do recorte TessCut (pixels)")
    ap.add_argument("--max_period", type=float, default=30.0, help="Periodo máximo (dias) para o BLS")
    ap.add_argument("--neg_per_pos", type=int, default=2, help="Negativos por positivo")
    ap.add_argument("--regen_all_list", action="store_true", help="Regerar all_windows.txt ao final")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outroot = Path(args.outroot).resolve()

    print("\n[CFG] outroot       :", outroot)
    print("[CFG] tics         :", args.tics)
    print("[CFG] max_sectors  :", args.max_sectors)
    print("[CFG] tesscut_size :", args.tesscut_size)
    print("[CFG] max_period   :", args.max_period)
    print("[CFG] neg_per_pos  :", args.neg_per_pos)
    print("[CFG] regen_list   :", args.regen_all_list)

    total_ok = 0
    total_fail = 0

    for tic in args.tics:
        ok, fail = process_tic(
            tic=tic,
            outroot=outroot,
            max_sectors=args.max_sectors,
            tesscut_size=args.tesscut_size,
            max_period=args.max_period,
            neg_per_pos=args.neg_per_pos
        )
        total_ok += ok
        total_fail += fail

    if args.regen_all_list:
        regenerate_all_list(outroot)

    print(f"\n[SUMMARY] TPFs processados: {total_ok} | Falhas: {total_fail}")


if __name__ == "__main__":
    main()
