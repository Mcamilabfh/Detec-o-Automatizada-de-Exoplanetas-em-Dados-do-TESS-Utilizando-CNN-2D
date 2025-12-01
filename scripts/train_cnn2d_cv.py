#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Treino + validação cruzada agrupada (TIC/sector) para o CNN2D.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from dataset_npz import NPZWindowsDataset
from build_cnn2d import build_model
from rescore_with_centroid import compute_shift_from_npz  # para shift direto do npz


def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def get_groups(ds):
    groups = []
    for i in range(len(ds)):
        _, _, _, g = ds[i]
        groups.append(g)
    return groups


def train_one_fold(train_files, val_files, args, fold_idx, device, outdir: Path):
    ds_tr = NPZWindowsDataset(args.data, file_list=train_files, train=True)
    ds_va = NPZWindowsDataset(args.data, file_list=val_files, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = build_model(in_ch=4).to(device)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_metrics = None
    best_state = None
    last_metrics = None
    last_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        for X, y, _, _ in dl_tr:
            X, y = X.to(device), y.unsqueeze(1).to(device)
            opt.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            loss_tr += loss.item() * X.size(0)
        loss_tr /= len(ds_tr)

        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for X, y, _, _ in dl_va:
                X = X.to(device)
                logits = model(X).cpu().numpy().ravel()
                p = 1 / (1 + np.exp(-logits))
                ps.extend(p); ys.extend(y.numpy().ravel())
        ys = np.array(ys); ps = np.array(ps)
        auc = roc_auc_score(ys, ps)
        ap  = average_precision_score(ys, ps)

        metrics = {"fold": fold_idx, "epoch": epoch, "loss_tr": float(loss_tr), "auc": float(auc), "ap": float(ap)}
        print(metrics)
        last_metrics = metrics
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_metrics = metrics
            # clone tensors to freeze best epoch (CPU)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    # salva best.pt e preds se solicitado
    fold_dir = outdir / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save({"model": best_state, "metrics": best_metrics, "last_metrics": last_metrics}, fold_dir / "best.pt")
    if args.report_last and last_state is not None:
        torch.save({"model": last_state, "metrics": last_metrics}, fold_dir / "last.pt")

    if args.save_preds:
        # reconstroi modelo escolhido (best por padr�o; last se report_last)
        state_for_preds = best_state if not args.report_last else last_state
        model.load_state_dict(state_for_preds, strict=True)
        model.eval()
        ys, ps, keys, files = [], [], [], []
        with torch.no_grad():
            for X, y, key, _ in DataLoader(ds_va, batch_size=args.batch_size, shuffle=False):
                X = X.to(device)
                logits = model(X).cpu().numpy().ravel()
                p = 1 / (1 + np.exp(-logits))
                ys.extend(y.numpy().ravel())
                ps.extend(p)
                keys.extend(list(key))
                files.extend([str(f) for f in key])
        preds_path = fold_dir / "preds.csv"
        with open(preds_path, "w", encoding="utf-8") as f:
            f.write("key,y_true,proba_pos\n")
            for k, yv, pv in zip(keys, ys, ps):
                f.write(f"{k},{int(yv)},{pv:.10f}\n")

        # calcula shift_px direto do npz (buscando pelo key)
        shifts = []
        for k in keys:
            candidates = list(Path(args.data).rglob(f"{k}.npz"))
            p = candidates[0] if candidates else None
            if p is None:
                shifts.append(np.nan)
            else:
                try:
                    shifts.append(compute_shift_from_npz(p))
                except Exception:
                    shifts.append(np.nan)
        import pandas as pd
        df = pd.DataFrame({"key": keys, "y_true": ys, "proba_pos": ps, "shift_px": shifts})
        df.to_csv(fold_dir / "preds_with_shift.csv", index=False)

    # salva lista de validação
    with open(fold_dir / "val_files.txt", "w", encoding="utf-8") as f:
        for p in val_files:
            f.write(f"{p}\n")

    return best_metrics if not args.report_last else last_metrics


def main():
    ap = argparse.ArgumentParser(description="Treino CNN2D com GroupKFold.")
    ap.add_argument("--data", required=True, help="Pasta raiz dos .npz (com pos/ e neg/)")
    ap.add_argument("--outdir", default="runs/exp_cv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save_preds", action="store_true", help="Salvar preds e shift_px do fold.")
    ap.add_argument("--report_last", action="store_true", help="Usar ultimo epoch para preds/metricas (compat. com execucoes antigas).")
    args = ap.parse_args()

    seed_everything(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ds_all = NPZWindowsDataset(args.data, train=True)
    files = ds_all.files
    groups = get_groups(ds_all)
    idxs = np.arange(len(ds_all))

    gkf = GroupKFold(n_splits=args.folds)
    fold_metrics = []
    for k, (tr, va) in enumerate(gkf.split(idxs, groups=groups), start=1):
        train_files = [files[i] for i in tr]
        val_files = [files[i] for i in va]
        m = train_one_fold(train_files, val_files, args, k, device=torch.device("cpu"), outdir=Path(args.outdir))
        fold_metrics.append(m)

    Path(args.outdir, "cv_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
    # resumo
    aucs = [m["auc"] for m in fold_metrics]
    aps  = [m["ap"] for m in fold_metrics]
    summary = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_ap": float(np.mean(aps)),
        "std_ap": float(np.std(aps)),
    }
    Path(args.outdir, "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
