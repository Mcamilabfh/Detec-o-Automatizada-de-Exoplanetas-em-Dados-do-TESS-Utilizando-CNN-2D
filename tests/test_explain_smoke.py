import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from projeto_toi.models import cnn2d
from projeto_toi.pipeline import explain


def test_gradcam_smoke(tmp_path: Path, monkeypatch):
    """Gera uma figura Grad-CAM com npz e checkpoint sinteticos."""
    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook")

    data_root = tmp_path / "data"
    pos_dir = data_root / "pos"
    pos_dir.mkdir(parents=True)

    cube = np.random.rand(3, 5, 5).astype(np.float32)
    np.savez(pos_dir / "sample_pos.npz", cube=cube)

    model = cnn2d.build_model(in_ch=4)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    preds_path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "file": [pos_dir / "sample_pos.npz"],
            "proba_pos": [0.9],
            "y_true": [1],
        }
    ).to_csv(preds_path, index=False)

    outdir = tmp_path / "out_cam"
    cfg = explain.ExplainConfig(
        data_dir=data_root,
        ckpt_path=ckpt_path,
        preds_csv=preds_path,
        outdir=outdir,
        top_k=1,
    )
    explain.generate_explanations(cfg)

    files = list(outdir.glob("cam_*.png"))
    assert files, "Nenhuma imagem Grad-CAM foi gerada"
