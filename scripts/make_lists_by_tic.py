from pathlib import Path
import argparse, re

def group_from_path(p: Path) -> str:
    """
    Tenta extrair TIC e Setor do nome do arquivo.
    Exemplos aceitos no stem:
      ...TIC_307210830_S14..., ...tic307210830-s14..., ...TIC307210830...
    """
    name = p.stem.lower()
    mt = re.search(r"tic[_\-]?(\d+)", name)
    ms = re.search(r"s(ec(tor)?)?[_\-]?(\d+)", name)
    if mt and ms:
        return f"TIC{mt.group(1)}_S{ms.group(4)}"
    if mt:
        return f"TIC{mt.group(1)}"
    return p.stem

def main(args):
    root = Path(args.data)
    files = sorted(root.rglob("*.npz"))
    tic_tag = f"TIC{args.tic}"

    test = [p for p in files if group_from_path(p).startswith(tic_tag)]
    train = [p for p in files if p not in test]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_files.txt").write_text("\n".join(map(str, train)))
    (out / f"test_{args.tic}.txt").write_text("\n".join(map(str, test)))

    print({
        "total": len(files),
        "train": len(train),
        "test": len(test),
        "lists_dir": str(out.resolve())
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta com os .npz (ex.: data/windows)")
    ap.add_argument("--tic", required=True, help="ex.: 307210830")
    ap.add_argument("--outdir", default="runs/exp_tic/lists")
    main(ap.parse_args())
