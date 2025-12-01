"""Treino e validacao do classificador 2D em formato funcional."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ProjectConfig, load_config
from ..datasets.npz_dataset import NPZWindowsDataset
from ..models.cnn2d import build_model
from ..utils.logging import info
from ..utils.paths import ensure_dirs
from ..utils.randomness import seed_everything


@dataclass
class TrainConfig:
    data_dir: Path
    outdir: Path
    epochs: int = 12
    batch_size: int = 16
    lr: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    patience: int = 6
    pos_weight: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False


def _make_scheduler(opt):
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    except Exception:
        class _NoOpScheduler:
            def step(self, *args, **kwargs):
                pass
        return _NoOpScheduler()


def _split_files(ds_all: NPZWindowsDataset, val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    groups = []
    for i in range(len(ds_all)):
        _, _, _, g = ds_all[i]
        groups.append(g)
    idxs = np.arange(len(ds_all))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(idxs, groups=groups))
    train_files = [ds_all.files[i] for i in train_idx]
    val_files = [ds_all.files[i] for i in val_idx]
    return train_files, val_files


def _train_epoch(model, dl, criterion, opt, device) -> float:
    model.train()
    loss_tr = 0.0
    for X, y, _, _ in tqdm(dl, desc="Train", leave=False, disable=True):
        X, y = X.to(device), y.unsqueeze(1).to(device)
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        loss_tr += loss.item() * X.size(0)
    return loss_tr / max(1, len(dl.dataset))


def _evaluate(model, dl, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y, _, _ in dl:
            X = X.to(device)
            logits = model(X).cpu().numpy().ravel()
            p = 1 / (1 + np.exp(-logits))
            ps.extend(p)
            ys.extend(y.numpy().ravel())
    return np.array(ys), np.array(ps)


def _predict_to_csv(model, dataset: NPZWindowsDataset, device, out_path: Path) -> Path:
    rows: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for idx, path in enumerate(dataset.files):
            X, y, key, group = dataset[idx]
            logits = model(X.unsqueeze(0).to(device)).cpu().numpy().ravel()[0]
            prob = 1 / (1 + np.exp(-logits))
            rows.append(
                {
                    "key": key,
                    "group": group,
                    "y_true": float(y.item()),
                    "proba_pos": float(prob),
                    "file": str(path),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def train_model(cfg: TrainConfig, project: ProjectConfig | None = None) -> Dict:
    project = project or load_config()
    seed_everything(cfg.seed)
    ensure_dirs([cfg.outdir])

    ds_all = NPZWindowsDataset(cfg.data_dir, train=True)
    if len(ds_all) == 0:
        raise SystemExit(f"Nenhum .npz encontrado em {cfg.data_dir}.")

    train_files, val_files = _split_files(ds_all, cfg.val_ratio, cfg.seed)

    splits_dir = cfg.outdir / "splits"
    ensure_dirs([splits_dir])
    Path(splits_dir / "train_files.txt").write_text("\n".join(map(str, train_files)))
    Path(splits_dir / "val_files.txt").write_text("\n".join(map(str, val_files)))

    ds_tr = NPZWindowsDataset(cfg.data_dir, file_list=train_files, train=True)
    ds_va = NPZWindowsDataset(cfg.data_dir, file_list=val_files, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_ch=4).to(device)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = _make_scheduler(opt)

    best_auc = 0.0
    bad = 0
    log_hist: List[Dict] = []
    checkpoints: Dict[str, Path] = {}
    best_path: Path | None = None

    for epoch in range(1, cfg.epochs + 1):
        loss_tr = _train_epoch(model, dl_tr, criterion, opt, device)
        ys, ps = _evaluate(model, dl_va, device)
        auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
        ap = average_precision_score(ys, ps)
        log = {"epoch": epoch, "loss_tr": float(loss_tr), "auc": float(auc), "ap": float(ap), "lr": opt.param_groups[0]["lr"]}
        log_hist.append(log)
        info(json.dumps(log))

        ckpt_path = cfg.outdir / f"model_epoch{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": log}, ckpt_path)
        checkpoints[f"epoch_{epoch}"] = ckpt_path
        scheduler.step(auc if not np.isnan(auc) else None)

        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            bad = 0
            best_path = cfg.outdir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": log}, best_path)
            checkpoints["best"] = best_path
        else:
            bad += 1
            if bad >= cfg.patience:
                info(f"Early stopping (sem melhora por {cfg.patience} epocas).")
                break

    with open(cfg.outdir / "train_log.json", "w") as f:
        json.dump(log_hist, f, indent=2)

    # usa melhor checkpoint (se existir) para gerar preds da validacao
    if best_path and best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    preds_path = cfg.outdir / "preds.csv"
    _predict_to_csv(model, ds_va, device, preds_path)

    return {"log": log_hist, "checkpoints": checkpoints, "config": asdict(cfg), "preds_path": str(preds_path)}


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Treino simples do modelo 2D.")
    parser.add_argument("--data", default="", help="Pasta raiz dos .npz (default=data/datasets/windows)")
    parser.add_argument("--outdir", default="", help="Pasta de saida (default=artifacts/runs/exp_01)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    ns = parser.parse_args(args=args)

    cfg = load_config()
    data_dir = Path(ns.data) if ns.data else cfg.paths.datasets / "windows"
    outdir = Path(ns.outdir) if ns.outdir else cfg.paths.runs / "exp_01"
    train_model(
        cfg=TrainConfig(
            data_dir=data_dir,
            outdir=outdir,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            lr=ns.lr,
            val_ratio=ns.val_ratio,
            seed=ns.seed,
            patience=ns.patience,
            pos_weight=ns.pos_weight,
            num_workers=ns.num_workers,
            pin_memory=ns.pin_memory,
        ),
        project=cfg,
    )


if __name__ == "__main__":
    cli()
