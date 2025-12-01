# scripts/make_group_folds.py
import argparse, re, os, random
from collections import defaultdict
from sklearn.model_selection import GroupKFold

def parse_group(p):
    # tenta achar TIC_xxx e sector_yy no caminho
    m_tic = re.search(r"TIC_(\d+)", p)
    m_sec = re.search(r"sector[_\\/-]?(\d+)", p, re.I)
    tic = m_tic.group(1) if m_tic else "UNK"
    sec = m_sec.group(1) if m_sec else "UNK"
    return f"{tic}_{sec}"

ap=argparse.ArgumentParser()
ap.add_argument("--files", required=True)    # lista balanceada (ex: all_windows_1to1.txt)
ap.add_argument("--outdir", required=True)   # pasta para salvar folds
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
a=ap.parse_args()

paths = [ln.strip() for ln in open(a.files, encoding="utf-8") if ln.strip()]
groups = [parse_group(p) for p in paths]

os.makedirs(a.outdir, exist_ok=True)
gkf = GroupKFold(n_splits=a.folds)
for k, (tr, va) in enumerate(gkf.split(paths, groups=groups), start=1):
    trf = os.path.join(a.outdir, f"train_fold{k}.txt")
    vaf = os.path.join(a.outdir, f"val_fold{k}.txt")
    with open(trf, "w", encoding="utf-8") as f:
        f.write("\n".join(paths[i] for i in tr))
    with open(vaf, "w", encoding="utf-8") as f:
        f.write("\n".join(paths[i] for i in va))
    print(f"[fold {k}] train={len(tr)} val={len(va)} grupos_train={len(set(groups[i] for i in tr))} grupos_val={len(set(groups[i] for i in va))}")
