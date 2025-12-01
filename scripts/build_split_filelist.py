# scripts/build_split_filelist.py
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files_txt", required=True, help="lista original usada no treino (um caminho .npz por linha)")
    ap.add_argument("--idx_txt", required=True, help="arquivo com índices (um por linha) ex: fold*/splits/val_idx.txt")
    ap.add_argument("--out_txt", required=True, help="para onde salvar a lista de arquivos selecionados")
    args = ap.parse_args()

    files = [ln.strip() for ln in Path(args.files_txt).read_text().splitlines() if ln.strip()]
    idxs  = [int(ln.strip()) for ln in Path(args.idx_txt).read_text().splitlines() if ln.strip()]
    sel   = [files[i] for i in idxs if 0 <= i < len(files)]

    out = Path(args.out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sel), encoding="utf-8")
    print(f"[ok] salvo {len(sel)} caminhos em {out}")

if __name__ == "__main__":
    main()
