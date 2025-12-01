# scripts/eval_cnn2d_ensemble_topk.py
import argparse, json, glob, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from dataset_npz import NPZWindowsDataset
from build_cnn2d import build_model

TARGET_HW = (15, 15)  # mesmo fixo usado no treino

def collate_pad(batch):
    xs, ys, keys = [], [], []
    max_h = max_w = 0
    for item in batch:
        if len(item) == 4:
            X, y, key, _ = item
        else:
            X, y = item[0], item[1]
            key = None
        xs.append(X); ys.append(y); keys.append(key)
        _, h, w = X.shape
        max_h = max(max_h, h); max_w = max(max_w, w)
    padded = []
    for X in xs:
        _, h, w = X.shape
        Xp = F.pad(X, (0, max_w - w, 0, max_h - h))
        padded.append(Xp)
    Xb = torch.stack(padded, 0)
    yb = torch.tensor(ys)
    return Xb, yb, keys

def load_val_metrics(fold_dir: Path, metric: str):
    vm = fold_dir / "val_metrics.json"
    if not vm.exists():
        return None
    d = json.loads(vm.read_text())
    if metric.lower() == "auc":
        return float(d.get("AUC", float("nan")))
    elif metric.lower() == "ap":
        return float(d.get("AP", float("nan")))
    return None

def pick_topk_ckpts(run_dir: Path, k: int, metric: str):
    folds = sorted([p for p in run_dir.glob("fold*") if p.is_dir()])
    scored = []
    for fd in folds:
        m = load_val_metrics(fd, metric)
        ckpt = fd / "best.pt"
        if m is not None and ckpt.exists() and not np.isnan(m):
            scored.append((m, ckpt))
    if not scored:
        raise SystemExit(f"Nenhum val_metrics.json válido em {run_dir}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max(1, k)]  # [(score, ckpt), ...]

def build_load_model(ckpt_path: Path, device: torch.device):
    model = build_model(in_ch=4).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@torch.no_grad()
def ensemble_predict(models, dl, device, weights=None, logits=False):
    ys_all, ps_all, keys_all = [], [], []
    if weights is None:
        weights = [1.0] * len(models)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / w.sum()

    for X, y, keys in tqdm(dl, desc="eval-ensemble-topk"):
        X = X.to(device)
        X = F.interpolate(X, size=TARGET_HW, mode="bilinear", align_corners=False)
        prob_sum = torch.zeros((X.size(0), 1), device=device)
        for mi, m in enumerate(models):
            out = m(X)               # (B,1)
            p = torch.sigmoid(out) if logits else out  # PROBABILIDADE
            prob_sum += w[mi] * p
        prob = prob_sum.clamp(0, 1).detach().cpu().numpy().ravel()

        ys_all.extend(y.numpy().ravel().tolist())
        ps_all.extend(prob.tolist())
        for k in keys:
            keys_all.append("" if k is None else str(k))

    return np.array(ys_all), np.array(ps_all), keys_all

def save_preds(outdir: Path, ys, ps, keys):
    outdir.mkdir(parents=True, exist_ok=True)
    # Compatibilidade com select_threshold.py: precisa 'y' e 'p' (ou 'p_mean')
    with open(outdir / "preds.csv", "w", encoding="utf-8") as f:
        f.write("key,y,p,p_mean\n")
        for k, y, p in zip(keys, ys, ps):
            f.write(f"{k},{int(y)},{p:.10f},{p:.10f}\n")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Coleta checkpoints
    ckpts = []
    if args.ckpt_glob:
        for p in glob.glob(args.ckpt_glob):
            if os.path.isfile(p) and p.lower().endswith(".pt"):
                ckpts.append(Path(p))

    weights = None
    if args.run_dir:
        top = pick_topk_ckpts(Path(args.run_dir), args.k, args.metric)
        print(f"[info] Top-{args.k} por {args.metric.upper()}:")
        for i, (score, c) in enumerate(top, 1):
            print(f"  {i:02d}. {c}  ({args.metric}={score:.3f})")
        if not ckpts:
            ckpts = [c for _, c in top]
        if args.weights == "metric":
            weights = [float(s) for s, _ in top[:len(ckpts)]]

    if not ckpts:
        raise SystemExit("Nenhum checkpoint encontrado. Use --ckpt_glob ou --run_dir.")

    print(f"[info] {len(ckpts)} checkpoints para ensemble (Top-K).")
    for i, c in enumerate(ckpts, 1):
        print(f"  {i:02d}. {c}")

    # Dataset
    files = [Path(x.strip()) for x in Path(args.val_files).read_text().splitlines() if x.strip()]
    ds = NPZWindowsDataset(args.data, train=False, file_list=files)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_pad)

    # Modelos
    models = [build_load_model(c, device) for c in ckpts]

    # Predição
    ys, ps, keys = ensemble_predict(models, dl, device, weights=weights, logits=args.logits)

    # Métricas
    auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
    ap  = average_precision_score(ys, ps)

    outdir = Path(args.outdir)
    save_preds(outdir, ys, ps, keys)
    with open(outdir / "metrics.json", "w") as f:
        json.dump({"AUC": float(auc), "AP": float(ap)}, f, indent=2)

    print({"AUC": float(auc), "AP": float(ap)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val_files", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    # Seleção Top-K
    ap.add_argument("--run_dir", default="", help="pasta com subpastas fold*/ contendo val_metrics.json e best.pt")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--metric", choices=["auc","ap"], default="auc")
    ap.add_argument("--weights", choices=["uniform","metric"], default="uniform")
    ap.add_argument("--ckpt_glob", default="")
    # Se o modelo retorna LOGITS, use --logits para aplicar sigmoid
    ap.add_argument("--logits", action="store_true", help="se definido, aplica sigmoid na saída do modelo")
    args = ap.parse_args()
    main(args)
