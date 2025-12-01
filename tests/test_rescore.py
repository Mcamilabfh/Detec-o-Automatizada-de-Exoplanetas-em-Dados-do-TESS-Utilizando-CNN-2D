import numpy as np
import pandas as pd
from pathlib import Path

from projeto_toi.pipeline import rescore as rsc


def test_compute_shift_from_npz_simple(tmp_path: Path):
    """Shift deve ser a distancia entre centroides (pre vs mediano)."""
    cube = np.zeros((5, 3, 3), dtype=float)
    cube[0, 0, 0] = 5.0
    cube[1, 0, 0] = 5.0  # pre-evento concentrado em (0,0)
    cube[2, 1, 1] = 1.0
    cube[3, 2, 2] = 8.0
    cube[4, 2, 2] = 8.0  # mediano ficara no centro geometrico (1,1)
    mask_before = np.array([1, 1, 0, 0, 0], dtype=int)

    path = tmp_path / "sample.npz"
    np.savez(path, cube=cube, mask_before=mask_before)

    shift = rsc.compute_shift_from_npz(path)
    assert np.isclose(shift, np.sqrt(2)), f"shift esperado sqrt(2), obtido {shift}"


def test_run_once_scores_and_metrics(tmp_path: Path):
    """Confirma score_final = p*exp(-tau*shift) e gravacao de CSV/JSON."""
    df = pd.DataFrame(
        {
            "key": ["a", "b"],
            "y_true": [0, 1],
            "proba_pos": [0.2, 0.8],
            "shift_px": [0.0, 1.0],
        }
    )
    out_path = tmp_path / "ranking.csv"
    summary = rsc.score_dataframe(df, p_col="proba_pos", y_col="y_true", tau=1.0, out_path=out_path)

    written = pd.read_csv(out_path).sort_values("key")
    expected_scores = df.assign(score_final=df["proba_pos"] * np.exp(-1.0 * df["shift_px"]))
    expected_scores = expected_scores.sort_values("key")
    assert np.allclose(written["score_final"], expected_scores["score_final"])
    assert out_path.with_suffix(".metrics.json").exists()
    assert "AUC" in summary and "AP" in summary
