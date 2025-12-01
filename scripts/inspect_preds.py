# scripts/inspect_preds.py
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, f1_score,
    average_precision_score, roc_auc_score
)

def load_preds_csv(p):
    ys, ps, files = [], [], []
    with open(p, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            raise SystemExit("preds.csv sem header")
        cols = {name:i for i,name in enumerate(header)}
        pcol = cols.get("p_mean", cols.get("p", None))
        if pcol is None or "y" not in cols or "file" not in cols:
            raise SystemExit("precisa de colunas: file, y e p (ou p_mean)")
        for row in r:
            files.append(row[cols["file"]])
            ys.append(float(row[cols["y"]]))
            ps.append(float(row[pcol]))
    return np.array(ys), np.array(ps), files

def summarize_split(name, y, p, thr):
    yhat = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y))>1 else float("nan")
    ap  = average_precision_score(y, p)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    prec = tp / max(1, (tp+fp))
    rec  = tp / max(1, (tp+fn))
    f1   = f1_score(y, yhat)
    return {
        "name": name, "threshold": thr,
        "AUC": float(auc), "AP": float(ap),
        "prec": float(prec), "rec": float(rec), "F1": float(f1),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

def best_thresholds(y, p, target_recall=None):
    # F1 max na curva PR
    prec, rec, thr_pr = precision_recall_curve(y, p)
    thr_pr = np.r_[0.0, thr_pr]  # alinhar tamanhos
    f1s = 2*prec*rec / np.maximum(prec+rec, 1e-9)
    i_f1 = int(np.nanargmax(f1s))
    thr_f1 = float(thr_pr[i_f1])

    # Youden (ROC)
    fpr, tpr, thr_roc = roc_curve(y, p)
    j = tpr - fpr
    i_j = int(np.argmax(j))
    thr_you = float(thr_roc[i_j])

    out = {
        "thr_f1": thr_f1,
        "F1_at_thr_f1": float(f1s[i_f1]),
        "thr_youden": thr_you,
        "YoudenJ_at_thr_youden": float(j[i_j]),
    }

    if target_recall is not None:
        # threshold mínimo que atinge recall >= alvo
        # percorre curva PR do maior para o menor threshold
        idxs = np.where(rec >= target_recall)[0]
        if len(idxs) > 0:
            thr_rec = float(thr_pr[idxs[0]])
        else:
            thr_rec = float(thr_pr[-1])  # mais permissivo
        out["thr_target_recall"] = thr_rec
        out["target_recall"] = float(target_recall)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds_csv", required=True)
    ap.add_argument("--holdout_preds_csv", required=True)
    ap.add_argument("--thr", type=float, default=None, help="threshold fixo; se não passar, usa thr_f1 do VAL")
    ap.add_argument("--target_recall", type=float, default=None, help="opcional: também reporta um threshold que atinja ≥ recall")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    yv, pv, _ = load_preds_csv(args.val_preds_csv)
    yh, ph, _ = load_preds_csv(args.holdout_preds_csv)

    picks = best_thresholds(yv, pv, target_recall=args.target_recall)
    thr_use = picks["thr_f1"] if args.thr is None else args.thr

    summary = {
        "thresholds": picks,
        "use_threshold": float(thr_use),
        "VAL": summarize_split("VAL", yv, pv, thr_use),
        "HOLDOUT": summarize_split("HOLDOUT", yh, ph, thr_use),
    }

    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out_json, "w"), indent=2)

if __name__ == "__main__":
    main()
