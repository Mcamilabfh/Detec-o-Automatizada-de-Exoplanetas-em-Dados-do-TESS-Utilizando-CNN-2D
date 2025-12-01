import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# tenta achar uma coluna pela 1ª que existir na lista
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_stem(s: str) -> str:
    s = str(s)
    name = s.split("/")[-1].split("\\")[-1]
    if "." in name:
        name = ".".join(name.split(".")[:-1])
    return name

def main(args):
    preds = pd.read_csv(args.preds)
    # normaliza chave do preds (stem)
    preds["key_norm"] = preds["key"].apply(to_stem)

    cen = pd.read_csv(args.centroid_csv)

    # descobrir nomes das colunas no CSV de centróide
    key_col = pick_col(cen, ["key", "filename", "file", "name"])
    shift_col = pick_col(cen, ["shift_px", "delta_cen_px", "delta_px", "delta", "centroid_shift", "centroid_px"])

    if key_col is None:
        raise SystemExit(
            f"Não encontrei coluna de chave no CSV de centróide. "
            f"Tente renomear para uma destas: ['key','filename','file','name'].\n"
            f"Colunas vistas: {list(cen.columns)}"
        )
    if shift_col is None:
        raise SystemExit(
            f"Não encontrei coluna de deslocamento (shift) no CSV. "
            f"Tente renomear para uma destas: ['shift_px','delta_cen_px','delta_px','delta','centroid_shift','centroid_px'].\n"
            f"Colunas vistas: {list(cen.columns)}"
        )

    # normaliza chave do centróide (stem)
    cen["key_norm"] = cen[key_col].apply(to_stem)

    # seleciona e junta
    shift = cen[["key_norm", shift_col]].rename(columns={shift_col: "shift_px"})
    df = preds.merge(shift, on="key_norm", how="left")

    # onde não houver centróide, assuma shift=0
    df["shift_px"] = df["shift_px"].fillna(0.0)

    # escore final: prob * exp(-tau * shift_px)
    tau = float(args.tau)
    df["score_final"] = df["proba_pos"] * np.exp(-tau * df["shift_px"])

    # ordenar e salvar
    cols_order = ["key", "y_true", "proba_pos", "shift_px", "score_final"]
    cols_order = [c for c in cols_order if c in df.columns]
    df.sort_values("score_final", ascending=False).to_csv(args.out, index=False, columns=cols_order)

    # resumo no console
    print({
        "rows": int(len(df)),
        "com_shift": int((df["shift_px"] > 0).sum()),
        "tau": tau,
        "out": str(Path(args.out).resolve())
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--centroid_csv", required=True)
    ap.add_argument("--out", default="runs/exp_01/ranking.csv")
    ap.add_argument("--tau", type=float, default=0.7)
    main(ap.parse_args())
