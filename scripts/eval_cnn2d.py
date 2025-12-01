# scripts/eval_cnn2d_ensemble.py
import argparse, os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from dataset_npz import NPZWindowsDataset
from build_cnn2d import build_model

TARGET_HW = (15, 15)

def collate_pad(batch):
    xs, ys, rest = [], [], []
    max_h = max_w = 0
    for item in batch:
        X, y, *r = item
        xs.append(X); ys.append(y); rest.append(r)
        _, h, w = X.shape
        if h > max_h: max_h = h
        if w > max_w: max_w = w
    padded = []
    for X in xs:
        _, h, w = X.shape
        Xp = F.pad(X, (0, max_w - w, 0, max_h - h))
        padded.append(Xp)
    Xb = torch.stack(padded, 0)
    yb = torch.tensor(ys)
    return Xb, yb, None, None

def load_checkpoints(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        p = Path(p)
        if not p.exists():
            print(f"[warn] checkpoint não encontrado: {p}")
            continue
        m = build_model(in_ch=4).to(device)
        ckpt = torch.load(p, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
    if not models:
        raise SystemExit("Nenhum checkpoint válido foi carregado.")
    return models

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # monta lista de arquivos do holdout
    lines = Path(args.val_files).read_text().splitlines()
    file_list = [Path(x.strip()) for x in lines if x.strip()]
    if len(file_list)==0:
        raise SystemExit(f"--val_files vazio: {args.val_files}")

    # dataset + dataloader
    ds = NPZWindowsDataset(args.data, train=False, file_list=file_list)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, collate_fn=collate_pad)

    # resolve ckpts: pelos argumentos diretos e/ou pelo glob
    ckpts = []
    if args.ckpt:
        ckpts.extend(args.ckpt)
    if args.ckpt_glob:
        ckpts.extend(sorted(glob.glob(args.ckpt_glob)))
    ckpts = list(dict.fromkeys(ckpts))  # remove duplicados mantendo ordem
    print(f"[info] {len(ckpts)} checkpoints para ensemble")
    for i, p in enumerate(ckpts, 1):
        print(f"  {i:02d}. {p}")

    models = load_checkpoints(ckpts, device)

    # inferência com média de probabilidades
    ys, ps_mean = [], []
    with torch.no_grad():
        for X, y, _, _ in tqdm(dl, desc="eval-ensemble"):
            X = X.to(device)
            X = F.interpolate(X, size=TARGET_HW, mode="bilinear", align_corners=False)
            # acumula probabilidades de cada modelo
            probs_sum = None
            for m in models:
                p = m(X).detach().cpu().numpy()
                probs_sum = p if probs_sum is None else (probs_sum + p)
            probs = (probs_sum / len(models)).ravel()
            ps_mean.extend(probs); ys.extend(y.numpy().ravel())

    ys = np.array(ys)
    ps_mean = np.array(ps_mean)
    auc = roc_auc_score(ys, ps_mean) if len(np.unique(ys))>1 else float("nan")
    ap  = average_precision_score(ys, ps_mean)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    json.dump({"AUC": float(auc), "AP": float(ap)}, open(outdir/"metrics.json","w"), indent=2)

    # salva CSV com previsões
    try:
        import csv
        with open(outdir/"preds.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file","y","p_mean"])
            for path, yv, pv in zip(file_list, ys.tolist(), ps_mean.tolist()):
                w.writerow([str(path), int(yv), float(pv)])
    except Exception:
        pass

    print({"AUC": float(auc), "AP": float(ap)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val_files", required=True, help="txt com caminhos .npz (1 por linha)")
    # use um ou vários --ckpt e/ou um --ckpt_glob
    ap.add_argument("--ckpt", action="append", help="checkpoint .pt (pode repetir a flag)")
    ap.add_argument("--ckpt_glob", default="", help="padrão glob para checkpoints, ex: runs/exp*/fold*/best.pt")
    ap.add_argument("--outdir", default="runs/eval_out_ensemble")
    ap.add_argument("--batch_size", type=int, default=64)
    main(ap.parse_args())
