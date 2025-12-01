"""Grad-CAM para interpretar as predicoes do modelo 2D."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ..datasets.npz_dataset import NPZWindowsDataset
from ..models.cnn2d import build_model
from ..utils.logging import info


def gradcam(model, x, target_layer):
    """Grad-CAM simples no target_layer (modulo conv)."""
    feats = []
    grads = []

    def f_hook(_, __, output):
        feats.append(output.detach())

    def b_hook(_, grad_input, grad_output):
        grads.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_backward_hook(b_hook)

    model.zero_grad()
    yhat = model(x)
    score = yhat.squeeze()
    score.backward()

    h1.remove()
    h2.remove()

    A = feats[-1]
    G = grads[-1]
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=False).squeeze(0)

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam.cpu().numpy()


def overlay_cam(img2d: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img2d, cmap="gray", interpolation="nearest")
    plt.imshow(cam, cmap="jet", alpha=alpha, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    return fig


@dataclass
class ExplainConfig:
    data_dir: Path
    ckpt_path: Path
    preds_csv: Path
    outdir: Path
    top_k: int = 12


def generate_explanations(cfg: ExplainConfig) -> List[Path]:
    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    info(f"Salvando explicacoes em: {outdir.resolve()}")

    ds = NPZWindowsDataset(cfg.data_dir, train=False)
    index_by_stem = {p.stem: i for i, p in enumerate(ds.files)}
    index_by_name = {p.name: i for i, p in enumerate(ds.files)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_ch=4).to(device)
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_layer = model.features[-1].net[0]

    df = pd.read_csv(cfg.preds_csv)
    score_col = "proba_pos" if "proba_pos" in df.columns else ("p_mean" if "p_mean" in df.columns else None)
    file_col = "file" if "file" in df.columns else ("path" if "path" in df.columns else None)
    if score_col is None or file_col is None:
        raise SystemExit("O CSV de preds precisa ter colunas de caminho (file/path) e 'proba_pos' ou 'p_mean'.")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top = df.head(int(cfg.top_k))

    saved = 0
    saved_paths: List[Path] = []
    for _, row in tqdm(top.iterrows(), total=len(top), desc="Grad-CAM", disable=True):
        key = Path(str(row[file_col])).stem
        idx = None
        if key in index_by_stem:
            idx = index_by_stem[key]
        elif key in index_by_name:
            idx = index_by_name[key]
        else:
            matches = [i for i, p in enumerate(ds.files) if key in p.stem or p.stem in key]
            if matches:
                idx = matches[0]
        if idx is None:
            info(f"Nao encontrei key='{key}' no dataset. Pulando.")
            continue

        X, y, key_ds, _ = ds[idx]
        base = X[0].numpy()
        base -= base.min()
        base /= (base.max() + 1e-8)
        x1 = X.unsqueeze(0).to(device)
        cam = gradcam(model, x1, target_layer)
        fig = overlay_cam(base, cam, alpha=0.45)
        out_path = outdir / f"cam_{key}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved += 1
        saved_paths.append(out_path)
        info(f"[OK] Salvo: {out_path.name}")

    if saved == 0:
        info("Nenhuma imagem foi salva. Verifique se as 'keys' do preds.csv batem com os nomes/stems dos .npz.")
    else:
        info(f"[DONE] {saved} figuras salvas em {outdir.resolve()}")
    return saved_paths


def cli(args=None) -> None:
    parser = argparse.ArgumentParser(description="Gera mapas Grad-CAM para o top-k.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--outdir", default="artifacts/runs/exp_01/explain")
    parser.add_argument("--top-k", type=int, default=12)
    ns = parser.parse_args(args=args)
    generate_explanations(
        ExplainConfig(
            data_dir=Path(ns.data),
            ckpt_path=Path(ns.ckpt),
            preds_csv=Path(ns.preds),
            outdir=Path(ns.outdir),
            top_k=ns.top_k,
        )
    )


if __name__ == "__main__":
    cli()
