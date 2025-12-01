#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, subprocess, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
MAKE_WINDOWS = HERE / "make_windows_auto_bls.py"

def count_pos_neg(win_dir: Path):
    if not win_dir.exists():
        return 0, 0
    c = Counter(("POS" if p.name.startswith("pos_") else "NEG") for p in win_dir.glob("*.npz"))
    return c.get("POS", 0), c.get("NEG", 0)

def run_make(outdir: Path, max_period: float, neg_per_pos: int):
    cmd = [sys.executable, str(MAKE_WINDOWS), "--outdir", str(outdir),
           "--max_period", str(max_period), "--neg_per_pos", str(neg_per_pos)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=False)

def regen_all_list(outroot: Path):
    runs_dir = outroot / "runs" / "exp_windows_auto" / "lists"
    runs_dir.mkdir(parents=True, exist_ok=True)
    all_list = runs_dir / "all_windows.txt"
    lines = []
    for npz in outroot.rglob("data/windows_auto/*.npz"):
        try:
            lines.append(str(npz.resolve()))
        except Exception:
            lines.append(str(npz))
    lines = sorted(set(lines))
    all_list.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Lista mestre regenerada: {all_list}  (total {len(lines)} janelas)")

def export_csv(outroot: Path):
    import csv
    csvp = outroot / "runs" / "exp_windows_auto" / "lists" / "windows_index.csv"
    csvp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for npz in outroot.rglob("data/windows_auto/*.npz"):
        tic    = npz.parents[3].name.replace("TIC_", "")
        sector = npz.parents[2].name.replace("sector_", "")
        klass  = "POS" if npz.name.startswith("pos_") else "NEG"
        rows.append((tic, sector, klass, str(npz)))
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TIC", "Sector", "Class", "File"])
        w.writerows(sorted(rows))
    print(f"[OK] CSV salvo: {csvp}  (linhas {len(rows)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default=".", help="Raiz do projeto")
    ap.add_argument("--target_ratio", type=float, default=2.0,
                    help="Meta de NEG >= POS * target_ratio (default=2.0)")
    ap.add_argument("--neg1", type=int, default=3, help="neg_per_pos 1ª tentativa")
    ap.add_argument("--neg2", type=int, default=4, help="neg_per_pos 2ª tentativa (fallback)")
    ap.add_argument("--max_period_fallback", type=float, default=20.0, help="max_period fallback")
    args = ap.parse_args()

    outroot = Path(args.outroot).resolve()
    sectors = sorted(outroot.glob("TIC_* /sector_*".replace(" ", "")))  # TIC_* / sector_*

    print(f"[INFO] Checando {len(sectors)} setores…")
    to_fix = []
    for sector_dir in sectors:
        win_dir = sector_dir / "data" / "windows_auto"
        pos, neg = count_pos_neg(win_dir)
        if pos == 0:
            continue  # nada para balancear
        ok = (neg >= pos * args.target_ratio)
        print(f"[CHECK] {sector_dir}: POS={pos} NEG={neg}  -> {'OK' if ok else 'FIX'}")
        if not ok:
            to_fix.append((sector_dir, pos, neg))

    # 1ª passagem de correção
    for sector_dir, pos, neg in to_fix:
        print(f"[FIX-1] {sector_dir}  (POS={pos} NEG={neg})  -> neg_per_pos={args.neg1}")
        run_make(sector_dir, max_period=30.0, neg_per_pos=args.neg1)

    # Recontar e preparar fallback nos setores que ainda não atingiram a meta
    still_bad = []
    for sector_dir, _, _ in to_fix:
        pos2, neg2 = count_pos_neg(sector_dir / "data" / "windows_auto")
        ok2 = (pos2 == 0) or (neg2 >= pos2 * args.target_ratio)
        print(f"[POST-1] {sector_dir}: POS={pos2} NEG={neg2}  -> {'OK' if ok2 else 'FALLBACK'}")
        if not ok2 and pos2 > 0:
            still_bad.append(sector_dir)

    # 2ª passagem (fallback: período menor + mais negativos)
    for sector_dir in still_bad:
        print(f"[FIX-2] {sector_dir}  -> max_period={args.max_period_fallback}  neg_per_pos={args.neg2}")
        run_make(sector_dir, max_period=args.max_period_fallback, neg_per_pos=args.neg2)

    regen_all_list(outroot)
    export_csv(outroot)
    print("[DONE] Balance fix completo.")

if __name__ == "__main__":
    main()
