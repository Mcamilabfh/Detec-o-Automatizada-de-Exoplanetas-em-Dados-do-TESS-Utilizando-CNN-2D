import os, sys, pathlib
from astropy.io import fits
from lightkurve import search_targetpixelfile
from lightkurve import TessTargetPixelFile
from astropy.utils.data import download_file
from astroquery.mast import Tesscut

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def save_tpf(obj, outpath, overwrite=True):
    if isinstance(obj, TessTargetPixelFile):
        # já é um objeto TPF carregado
        obj.to_fits(outpath, overwrite=overwrite)
    else:
        # é um caminho/HDUL
        if isinstance(obj, str):
            hdul = fits.open(obj)
        else:
            hdul = obj
        hdul.writeto(outpath, overwrite=overwrite)
        hdul.close()

def via_search_tpf(tic, sector):
    res = search_targetpixelfile(f"TIC {tic}", mission="TESS", sector=sector)
    if len(res) == 0:
        return None
    tpf = res.download()
    return tpf

def via_tesscut(tic, sector, size=11):
    # baixa um cutout “target pixel file”-like
    # retorna HDUList adequado para .writeto
    cutouts = Tesscut.get_cutouts(f"TIC {tic}", size=size, sector=sector)
    if cutouts is None or len(cutouts) == 0:
        return None
    # cada item é (HDUList, meta)
    # escolhemos o primeiro
    hdu, _ = cutouts[0]
    return hdu

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", required=True, type=str)
    ap.add_argument("--sector", required=True, type=int)
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--outroot", type=str, default=".")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tic = args.tic
    sector = args.sector
    outdir = os.path.join(args.outroot, f"data/TIC_{tic}/sector_{sector:02d}/data")
    ensure_dir(outdir)
    outpath = os.path.join(outdir, "tpf.fits")

    # 1) tenta TPF oficial
    tpf = via_search_tpf(tic, sector)
    if tpf is not None:
        save_tpf(tpf, outpath, overwrite=args.overwrite)
        print(f"[OK] TPF salvo: {outpath}")
        return

    # 2) fallback: Tesscut
    hdu = via_tesscut(tic, sector, size=args.size)
    if hdu is not None:
        save_tpf(hdu, outpath, overwrite=args.overwrite)
        print(f"[OK] Tesscut salvo: {outpath}")
        return

    print(f"[WARN] Sem TPF/Tesscut para TIC {tic} setor {sector}")

if __name__ == "__main__":
    main()
