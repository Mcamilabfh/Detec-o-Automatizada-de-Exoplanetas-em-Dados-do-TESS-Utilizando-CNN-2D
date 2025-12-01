# scripts/train_cnn2d_kfold_strat.py
import argparse, os, json, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from dataset_npz import NPZWindowsDataset
from build_cnn2d import build_model

# ---------- Config de entrada fixa ----------
TARGET_HW = (15, 15)  # (H, W) fixo para a rede

# ---------- Collate que faz padding por batch ----------
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
        Xp = F.pad(X, (0, max_w - w, 0, max_h - h))  # (left,right,top,bottom)
        padded.append(Xp)
    Xb = torch.stack(padded, 0)
    yb = torch.tensor(ys)
    return Xb, yb, None, None

# ---------- Utilidades ----------
def seed_all(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); os.environ["PYTHONHASHSEED"]=str(s)

class _NoOp:
    def step(self,*a,**k): pass

def make_scheduler(opt):
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    except Exception:
        return _NoOp()

def _group_labels(groups, ys):
    g2idx = {}
    for i, g in enumerate(groups):
        g2idx.setdefault(g, []).append(i)
    uniq_groups = np.array(list(g2idx.keys()), dtype=object)
    y_group = np.zeros(len(uniq_groups), dtype=int)
    for gi, g in enumerate(uniq_groups):
        idxs = g2idx[g]
        y_group[gi] = int(np.max(ys[idxs]))
    return uniq_groups, y_group, g2idx

def _make_group_stratified_splits(groups, ys, n_splits=5, seed=42):
    uniq_groups, y_group, g2idx = _group_labels(groups, ys)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    X_dummy = np.zeros_like(y_group)
    for g_tr, g_val in skf.split(X_dummy, y_group):
        tr_idxs, va_idxs = [], []
        for gi in g_tr:
            tr_idxs.extend(g2idx[uniq_groups[gi]])
        for gi in g_val:
            va_idxs.extend(g2idx[uniq_groups[gi]])
        splits.append((np.array(tr_idxs), np.array(va_idxs)))
    return splits

def _subset_labels(ds_subset):
    """Extrai labels (0/1) de um Subset."""
    ys = []
    for i in range(len(ds_subset)):
        y = ds_subset[i][1]
        yi = int(float(y.item()) if hasattr(y, "item") else y)
        ys.append(yi)
    return np.array(ys, dtype=int)

def _make_weighted_sampler(ds_subset):
    """Cria WeightedRandomSampler para balancear classes no treino."""
    ys = _subset_labels(ds_subset)
    counts = np.bincount(ys, minlength=2)
    # evita div por zero
    w_per_class = np.array([1.0 / c if c > 0 else 0.0 for c in counts], dtype=float)
    weights = [w_per_class[y] for y in ys]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, counts.tolist()

# ---------- Treino por fold ----------
def run_fold(fold, train_idx, val_idx, args, outdir, ds_all):
    ds_tr = Subset(ds_all, train_idx)
    ds_va = Subset(ds_all, val_idx)

    # DataLoader de validação (sem sampler)
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_pad
    )

    # DataLoader de treino: com ou sem sampler
    if args.use_weighted_sampler:
        sampler, counts = _make_weighted_sampler(ds_tr)
        print(f"[fold {fold}] class_counts(train)={counts} -> usando WeightedRandomSampler")
        dl_tr = DataLoader(
            ds_tr, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            num_workers=0, pin_memory=False, collate_fn=collate_pad
        )
    else:
        dl_tr = DataLoader(
            ds_tr, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, collate_fn=collate_pad
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_ch=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == "bce_logits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()
    scheduler = make_scheduler(opt)

    fold_dir = outdir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "splits").mkdir(exist_ok=True)
    Path(fold_dir/"splits"/"train_idx.txt").write_text("\n".join(map(str,train_idx)))
    Path(fold_dir/"splits"/"val_idx.txt").write_text("\n".join(map(str,val_idx)))

    best_auc, bad, hist = 0.0, 0, []

    for epoch in range(1, args.epochs+1):
        model.train(); loss_tr=0.0; n_tr=0
        for X,y,_,_ in tqdm(dl_tr, desc=f"[fold {fold}] train {epoch}/{args.epochs}"):
            X,y = X.to(device), y.unsqueeze(1).to(device)
            # resize para tamanho fixo
            X = F.interpolate(X, size=TARGET_HW, mode="bilinear", align_corners=False)

            opt.zero_grad()
            out = model(X)
            if args.loss == "bce_logits":
                loss = criterion(out, y)  # out = logits
            else:
                loss = criterion(out, y)  # out = probs (com Sigmoid no modelo)
            loss.backward()
            opt.step()

            loss_tr += loss.item()*X.size(0); n_tr += X.size(0)
        loss_tr /= max(n_tr,1)

        # ---- validação (durante treino) ----
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for X,y,_,_ in dl_va:
                X = X.to(device)
                X = F.interpolate(X, size=TARGET_HW, mode="bilinear", align_corners=False)
                out = model(X)
                if args.loss == "bce_logits":
                    prob = torch.sigmoid(out).cpu().numpy().ravel()
                else:
                    prob = out.detach().cpu().numpy().ravel()
                ps.extend(prob); ys.extend(y.numpy().ravel())
        ys, ps = np.array(ys), np.array(ps)
        auc = roc_auc_score(ys, ps) if len(np.unique(ys))>1 else float("nan")
        ap  = average_precision_score(ys, ps)

        hist.append({
            "epoch": epoch,
            "loss_tr": float(loss_tr),
            "auc": float(auc),
            "ap": float(ap),
            "lr": opt.param_groups[0]["lr"]
        })
        json.dump(hist, open(fold_dir/"train_log.json","w"), indent=2)
        torch.save({"model":model.state_dict(),"epoch":epoch}, fold_dir/f"model_epoch{epoch:03d}.pt")
        # scheduler usa métrica de validação (AUC)
        try:
            scheduler.step(auc)
        except Exception:
            pass

        if (not np.isnan(auc)) and (auc > best_auc + 1e-4):
            best_auc=auc; bad=0
            torch.save({"model":model.state_dict(),"epoch":epoch}, fold_dir/"best.pt")
        else:
            bad+=1
            if bad>=args.patience:
                print(f"[fold {fold}] early stop"); break

    # ---- avaliação final do fold ----
    ckpt=torch.load(fold_dir/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for X,y,_,_ in DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False, collate_fn=collate_pad):
            X = X.to(device)
            X = F.interpolate(X, size=TARGET_HW, mode="bilinear", align_corners=False)
            out = model(X)
            if args.loss == "bce_logits":
                prob = torch.sigmoid(out).cpu().numpy().ravel()
            else:
                prob = out.detach().cpu().numpy().ravel()
            ps.extend(prob); ys.extend(y.numpy().ravel())
    ys, ps = np.array(ys), np.array(ps)
    auc = roc_auc_score(ys, ps) if len(np.unique(ys))>1 else float("nan")
    ap  = average_precision_score(ys, ps)
    json.dump({"AUC":float(auc),"AP":float(ap)}, open(fold_dir/"val_metrics.json","w"), indent=2)
    return auc, ap

# ---------- Main ----------
def main(args):
    seed_all(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    file_list = None
    if args.files_txt:
        lines = Path(args.files_txt).read_text().splitlines()
        file_list = [Path(x.strip()) for x in lines if x.strip()]
        if len(file_list)==0:
            raise SystemExit(f"--files_txt vazio: {args.files_txt}")

    ds_all = NPZWindowsDataset(args.data, train=True, file_list=file_list)
    if len(ds_all)==0:
        raise SystemExit("Sem dados em --data / --files_txt.")

    # extrai grupos e labels por amostra para splits estratificados por grupo
    groups, ys = [], []
    for i in range(len(ds_all)):
        _, y, _, g = ds_all[i]
        groups.append(g); ys.append(int(float(y.item()) if hasattr(y,"item") else y))
    groups = np.array(groups, dtype=object)
    ys = np.array(ys, dtype=int)

    # stats por grupo
    def _group_labels(groups, ys):
        g2idx = {}
        for i,g in enumerate(groups): g2idx.setdefault(g,[]).append(i)
        uniq = np.array(list(g2idx.keys()), dtype=object)
        y_g = np.zeros(len(uniq), dtype=int)
        for j,g in enumerate(uniq): y_g[j] = int(np.max(ys[g2idx[g]]))
        return uniq, y_g, g2idx
    uniq_g, y_group, _ = _group_labels(groups, ys)
    print({"groups_total": len(uniq_g), "groups_pos": int((y_group==1).sum()), "groups_neg": int((y_group==0).sum())})
    if (y_group==1).sum()==0 or (y_group==0).sum()==0:
        warnings.warn("Algum fold pode ficar sem uma das classes em nível de grupo.")

    splits = _make_group_stratified_splits(groups, ys, n_splits=args.folds, seed=args.seed)

    aucs, aps = [], []
    for fold, (tr, va) in enumerate(splits, start=1):
        a, p = run_fold(fold, tr, va, args, outdir, ds_all)
        aucs.append(a); aps.append(p)
        print(f"[fold {fold}] AUC={a:.3f} AP={p:.3f}")

    summ = {"folds":args.folds,
            "AUC_mean":float(np.nanmean(aucs)), "AUC_std":float(np.nanstd(aucs)),
            "AP_mean":float(np.nanmean(aps)),   "AP_std":float(np.nanstd(aps))}
    json.dump(summ, open(outdir/"kfold_summary.json","w"), indent=2)
    print(summ)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta base (a raiz dos caminhos nos .txt)")
    ap.add_argument("--files_txt", default="", help="lista de arquivos .npz (um por linha). Se vazio, usa --data")
    ap.add_argument("--outdir", default="runs/exp_kfold_strat")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    # Novas flags:
    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="Usa WeightedRandomSampler no treino para balancear classes")
    ap.add_argument("--loss", choices=["bce","bce_logits"], default="bce",
                    help="bce: modelo retorna probabilidade (Sigmoid interno). bce_logits: modelo retorna logits.")
    args = ap.parse_args()
    main(args)
