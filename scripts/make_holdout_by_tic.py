# scripts/make_holdout_by_tic.py
from pathlib import Path
import argparse, numpy as np

def has_tic(npz_path: Path, tic_target: str) -> bool:
    try:
        z = np.load(npz_path, allow_pickle=True)
        meta = z.get("meta")
        if meta is None:
            return False
        m = meta.item() if hasattr(meta, "item") else meta
        tic = str(m.get("tic") or m.get("TIC") or m.get("target_id") or "")
        return tic == str(tic_target)
    except Exception:
        return False

def main(args):
        root = Path(args.data)
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

        all_npz = sorted(root.rglob("*.npz"))
        test_only = []
        train_except = []

        for p in all_npz:
            if has_tic(p, args.tic):
                test_only.append(str(p))
            else:
                train_except.append(str(p))

        (outdir / f"test_only_{args.tic}.txt").write_text("\n".join(test_only))
        (outdir / f"train_except_{args.tic}.txt").write_text("\n".join(train_except))

        print({
            "total": len(all_npz),
            "train_except": len(train_except),
            "test_only": len(test_only),
            "lists_dir": str(outdir.resolve())
        })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta raiz onde estão os .npz (ex: data/windows)")
    ap.add_argument("--tic", required=True, help="ex: 307210830")
    ap.add_argument("--outdir", required=True, help="ex: runs/exp_tic307210830/lists")
    main(ap.parse_args())
