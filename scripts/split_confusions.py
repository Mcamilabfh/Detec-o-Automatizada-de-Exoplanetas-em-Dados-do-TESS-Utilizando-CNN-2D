# scripts/split_confusions.py
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
import json

def load_preds(csv_path: str):
    df = pd.read_csv(csv_path)

    cols = set(c.lower() for c in df.columns)

    # coluna de arquivo: aceita 'file', 'key' ou 'path'
    fcol = None
    for cand in ["file", "key", "path"]:
        if cand in cols:
            fcol = cand
            break

    # coluna de rótulo
    ycol = "y" if "y" in cols else ("label" if "label" in cols else None)

    # coluna de probabilidade (ou média do ensemble)
    pcol = None
    for cand in ["p_mean", "p", "prob"]:
        if cand in cols:
            pcol = cand
            break

    assert fcol is not None and ycol is not None and pcol is not None, \
        "precisa de colunas file/key/path, y/label e p_mean/p/prob"

    files = df[[c for c in df.columns if c.lower() == fcol][0]].astype(str).tolist()
    y = df[[c for c in df.columns if c.lower() == ycol][0]].astype(int).to_numpy()
    p = df[[c for c in df.columns if c.lower() == pcol][0]].astype(float).to_numpy()
    return files, y, p

def write_list(lst_path: Path, items):
    lst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lst_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(str(it) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True, help="CSV com colunas file/key, y e p/p_mean/prob")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    files, y, p = load_preds(args.preds_csv)
    thr = float(args.threshold)

    yhat = (p >= thr).astype(int)

    TP_idx = np.where((y == 1) & (yhat == 1))[0]
    FP_idx = np.where((y == 0) & (yhat == 1))[0]
    TN_idx = np.where((y == 0) & (yhat == 0))[0]
    FN_idx = np.where((y == 1) & (yhat == 0))[0]

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # salva listas .txt
    write_list(outdir / "TP.txt", [files[i] for i in TP_idx])
    write_list(outdir / "FP.txt", [files[i] for i in FP_idx])
    write_list(outdir / "TN.txt", [files[i] for i in TN_idx])
    write_list(outdir / "FN.txt", [files[i] for i in FN_idx])

    # também um resumo em JSON
    summary = {
        "threshold": thr,
        "counts": {
            "TP": int(len(TP_idx)),
            "FP": int(len(FP_idx)),
            "TN": int(len(TN_idx)),
            "FN": int(len(FN_idx)),
        }
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print({"TP": len(TP_idx), "FP": len(FP_idx), "TN": len(TN_idx), "FN": len(FN_idx)})

if __name__ == "__main__":
    main()
