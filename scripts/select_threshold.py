# scripts/select_threshold.py
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, average_precision_score,
    roc_auc_score
)

def load_preds_csv(p):
    ys, ps = [], []
    with open(p, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            raise SystemExit("preds.csv sem header")
        cols = {name:i for i,name in enumerate(header)}
        pcol = cols.get("p_mean", cols.get("p", None))
        if pcol is None or "y" not in cols:
            raise SystemExit("preds.csv precisa ter colunas 'y' e 'p' (ou 'p_mean')")
        ycol = cols["y"]
        for row in r:
            ys.append(float(row[ycol]))
            ps.append(float(row[pcol]))
    return np.array(ys), np.array(ps)

def pick_by_fbeta(y, p, beta=1.0):
    prec, rec, thr = precision_recall_curve(y, p)
    thr = np.r_[0.0, thr]
    beta2 = beta*beta
    fbeta = (1+beta2) * prec * rec / np.maximum(beta2*prec + rec, 1e-9)
    idx = int(np.nanargmax(fbeta))
    return float(thr[idx]), float(fbeta[idx])

def pick_by_f1(y, p):
    return pick_by_fbeta(y, p, beta=1.0)

def pick_by_youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(j[idx])

def pick_by_targets(y, p, target_recall=None, target_precision=None):
    prec, rec, thr = precision_recall_curve(y, p)
    thr = np.r_[0.0, thr]
    # por padrão, thresholds descem conforme recall sobe
    cand_idxs = np.arange(len(thr))
    if target_recall is not None:
        cand_idxs = cand_idxs[rec >= target_recall]
    if target_precision is not None:
        cand_idxs = cand_idxs[prec >= target_precision] if target_recall is None else cand_idxs[prec[cand_idxs] >= target_precision]
    if len(cand_idxs) == 0:
        # se não dá pra atingir as metas, escolhe o mais permissivo (último)
        idx = len(thr)-1
    else:
        # pega o MENOR threshold que atende (mais conservador que mantém as metas)
        idx = int(cand_idxs[0])
    return float(thr[idx]), {
        "precision_at_thr": float(prec[idx]),
        "recall_at_thr": float(rec[idx])
    }

def summarize(y, p, thr):
    yhat = (p >= thr).astype(float)
    auc = roc_auc_score(y, p) if len(np.unique(y))>1 else float("nan")
    ap  = average_precision_score(y, p)
    tp = int(np.sum((y==1) & (yhat==1)))
    fp = int(np.sum((y==0) & (yhat==1)))
    tn = int(np.sum((y==0) & (yhat==0)))
    fn = int(np.sum((y==1) & (yhat==0)))
    prec = tp / max(1, tp+fp)
    rec  = tp / max(1, tp+fn)
    f1   = 2*prec*rec / max(1e-9, prec+rec)
    return {
        "threshold": float(thr),
        "AUC": float(auc), "AP": float(ap),
        "prec": float(prec), "rec": float(rec), "F1": float(f1),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_preds_csv", required=True)
    ap.add_argument("--holdout_preds_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--opt", choices=["f1","youden","fbeta","targets"], default="f1")
    ap.add_argument("--beta", type=float, default=0.5, help="para fbeta (beta<1 puxa precisão; >1 puxa recall)")
    ap.add_argument("--target_recall", type=float, default=None)
    ap.add_argument("--target_precision", type=float, default=None)
    args = ap.parse_args()

    yv, pv = load_preds_csv(args.val_preds_csv)
    yh, ph = load_preds_csv(args.holdout_preds_csv)

    details = {}
    if args.opt == "f1":
        thr_star, score = pick_by_f1(yv, pv)
        details["F1_val"] = score
    elif args.opt == "youden":
        thr_star, score = pick_by_youden(yv, pv)
        details["YoudenJ_val"] = score
    elif args.opt == "fbeta":
        thr_star, score = pick_by_fbeta(yv, pv, beta=args.beta)
        details[f"F{args.beta}_val"] = score
        details["beta"] = args.beta
    else:
        thr_star, tr = pick_by_targets(yv, pv, args.target_recall, args.target_precision)
        details.update({
            "target_recall": args.target_recall,
            "target_precision": args.target_precision,
            **tr
        })

    out = {
        "criterion": args.opt,
        "thr_star": float(thr_star),
        "VAL": summarize(yv, pv, thr_star),
        "HOLDOUT": summarize(yh, ph, thr_star),
        "details": details
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
