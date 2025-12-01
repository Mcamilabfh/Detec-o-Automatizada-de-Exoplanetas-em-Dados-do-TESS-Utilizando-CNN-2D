import argparse, os, sys, numpy as np

def is_valid_npz(p):
    try:
        z = np.load(p, allow_pickle=True)
        if "cube" not in z or "time_btjd" not in z:
            return False, "missing keys"
        cube = z["cube"]; t = z["time_btjd"]
        if not hasattr(cube, "ndim") or cube.ndim != 3: return False, "cube shape"
        if cube.shape[0] < 3: return False, "too few frames"
        if np.any(~np.isfinite(cube)): return False, "non-finite"
        if len(t) != cube.shape[0]: return False, "time len mismatch"
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=False, help="lista com caminhos de .npz (um por linha)")
    ap.add_argument("--scan", nargs="*", help="pastas para varrer em busca de .npz")
    ap.add_argument("--out", required=True, help="arquivo de saída com apenas .npz válidos")
    args = ap.parse_args()

    paths = []
    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s: continue
                paths.append(s)
    if args.scan:
        for root in args.scan:
            for dp,_,files in os.walk(root):
                for f in files:
                    if f.lower().endswith(".npz"):
                        paths.append(os.path.join(dp,f))

    seen=set(); paths=[p for p in paths if not (p in seen or seen.add(p))]
    good=[]; bad=[]
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        ok,why = is_valid_npz(p)
        if ok: good.append(p)
        else:  bad.append((p,why))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in good: f.write(p+"\n")

    print(f"[SUMMARY] total={len(paths)}  good={len(good)}  bad={len(bad)}")
    if bad:
        bad_log = os.path.splitext(args.out)[0] + "_bad.txt"
        with open(bad_log,"w",encoding="utf-8") as f:
            for p,why in bad: f.write(f"{p}\t{why}\n")
        print(f"[INFO] lista de ruins: {bad_log}")

if __name__ == "__main__":
    main()
