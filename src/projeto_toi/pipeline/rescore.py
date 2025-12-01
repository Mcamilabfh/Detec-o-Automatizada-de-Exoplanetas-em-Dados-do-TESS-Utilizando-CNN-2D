"""Re-score usando deslocamento de centroide para penalizar candidatos off-target."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import ProjectConfig, load_config
from ..utils.logging import info


def _centroid_from_frame(frame: np.ndarray) -> tuple[float, float]:
    H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W]
    flux = frame - frame.min()
    flux = np.clip(flux, 0, None)
    total = flux.sum()
    if total <= 0:
        return (W - 1) / 2.0, (H - 1) / 2.0
    cx = float((flux * xx).sum() / total)
    cy = float((flux * yy).sum() / total)
    return cx, cy


def compute_shift_from_npz(npz_path: Path) -> float:
    """Distancia entre centroide pre-evento e centroide mediano."""
    z = np.load(npz_path, allow_pickle=True)
    cube = z["cube"]
    T, H, W = cube.shape

    med = np.nanmedian(cube, axis=0)
    cx_med, cy_med = _centroid_from_frame(med)

    mask_before = z.get("mask_before")
    if mask_before is not None and len(mask_before) == T and np.sum(mask_before) > 0:
        pre_frame = np.nanmean(cube[np.array(mask_before, dtype=bool)], axis=0)
    else:
        k = max(1, int(0.2 * T))
        pre_frame = np.nanmean(cube[:k], axis=0)
    cx_pre, cy_pre = _centroid_from_frame(pre_frame)
    shift = float(np.hypot(cx_pre - cx_med, cy_pre - cy_med))
    return shift


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_stem(s: str) -> str:
    s = str(s)
    name = s.split("/")[-1].split("\\")[-1]
    if "." in name:
        name = ".".join(name.split(".")[:-1])
    return name


@dataclass
class RescoreConfig:
    tau_list: List[float]
    preds_csv: Path
    centroid_csv: Optional[Path] = None
    data_root: Optional[Path] = None
    out_path: Path = Path("ranking.csv")


def _attach_shift(df: pd.DataFrame, data_root: Optional[Path], centroid_csv: Optional[Path]) -> pd.DataFrame:
    if centroid_csv:
        cen = pd.read_csv(centroid_csv)
        key_col = _pick_col(cen, ["key", "filename", "file", "name"])
        shift_col = _pick_col(cen, ["shift_px", "delta_cen_px", "delta_px", "delta", "centroid_shift", "centroid_px"])
        if key_col is None or shift_col is None:
            raise SystemExit("CSV de centroide precisa ter chave (key/filename/...) e deslocamento (shift_px/...).")
        cen["key_norm"] = cen[key_col].apply(_to_stem)
        shift = cen[["key_norm", shift_col]].rename(columns={shift_col: "shift_px"})
        df["key_norm"] = df["key"].apply(_to_stem)
        df = df.merge(shift, on="key_norm", how="left")
    else:
        file_col = "file" if "file" in df.columns else "key"
        shifts = []
        for fp in df[file_col]:
            p = Path(fp)
            if not p.exists() and data_root:
                maybe = list(Path(data_root).rglob(f"{p.stem}.npz"))
                p = maybe[0] if maybe else None
            if p is None or not p.exists():
                shifts.append(np.nan)
            else:
                try:
                    shifts.append(compute_shift_from_npz(p))
                except Exception:
                    shifts.append(np.nan)
        df["shift_px"] = shifts
    df["shift_px"] = df["shift_px"].fillna(0.0)
    return df


def _score_once(df: pd.DataFrame, p_col: str, y_col: str, tau: float, out_path: Path) -> Dict:
    df = df.copy()
    df["score_final"] = df[p_col] * np.exp(-tau * df["shift_px"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["key", y_col, p_col, "shift_px", "score_final"]
    df.sort_values("score_final", ascending=False).to_csv(out_path, index=False, columns=cols)
    y_true = df[y_col].to_numpy()
    s_final = df["score_final"].to_numpy()
    auc = roc_auc_score(y_true, s_final) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, s_final)
    summary = {"AUC": float(auc), "AP": float(ap), "rows": int(len(df)), "tau": tau, "out": str(out_path)}
    with open(out_path.with_suffix(".metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    info(json.dumps(summary))
    return summary


def score_dataframe(df: pd.DataFrame, p_col: str, y_col: str, tau: float, out_path: Path) -> Dict:
    """API publica para re-score de um DataFrame ja com shift."""
    return _score_once(df, p_col, y_col, tau, out_path)


def rescore_predictions(cfg: RescoreConfig, project: ProjectConfig | None = None) -> List[Dict]:
    project = project or load_config()
    preds = pd.read_csv(cfg.preds_csv)
    preds["key_norm"] = preds["key"].apply(_to_stem)
    p_col = _pick_col(preds, ["proba_pos", "p_mean", "p"])
    if p_col is None:
        raise SystemExit("Preds precisa ter coluna de prob: proba_pos / p_mean / p.")
    y_col = _pick_col(preds, ["y_true", "y"])
    if y_col is None:
        raise SystemExit("Preds precisa ter coluna de label: y_true / y.")
    preds = _attach_shift(preds, cfg.data_root, cfg.centroid_csv)

    summaries = []
    for tau in cfg.tau_list:
        suffix = f"_tau{tau}".replace(".", "p") if len(cfg.tau_list) > 1 else ""
        out_path = cfg.out_path
        if suffix:
            out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        summaries.append(_score_once(preds, p_col, y_col, tau, out_path))
    if len(summaries) > 1:
        best = max(summaries, key=lambda x: x["AUC"])
        info(f"[BEST AUC] {json.dumps(best)}")
    return summaries


def cli(args=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Re-score preditor penalizando deslocamento de centroide.")
    parser.add_argument("--preds", required=True, help="CSV com key,y,p/proba_pos/p_mean")
    parser.add_argument("--centroid-csv", default="", help="CSV com shift_px (opcional).")
    parser.add_argument("--data-root", default="", help="Raiz para buscar .npz se 'key' nao for caminho completo.")
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--tau-list", default="", help="Lista de tau separados por virgula; sobrescreve --tau.")
    parser.add_argument("--out", default="ranking.csv")
    ns = parser.parse_args(args=args)

    tau_list = [float(t) for t in ns.tau_list.split(",")] if ns.tau_list else [float(ns.tau)]
    cfg_obj = load_config()
    data_root = Path(ns.data_root) if ns.data_root else cfg_obj.paths.datasets / "windows"
    cfg = RescoreConfig(
        tau_list=tau_list,
        preds_csv=Path(ns.preds),
        centroid_csv=Path(ns.centroid_csv) if ns.centroid_csv else None,
        data_root=data_root,
        out_path=Path(ns.out),
    )
    rescore_predictions(cfg)


if __name__ == "__main__":
    cli()
