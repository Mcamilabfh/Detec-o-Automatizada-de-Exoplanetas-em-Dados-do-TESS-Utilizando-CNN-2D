#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
consolidate_windows.py
----------------------
Varre *.npz de janelas (pos/neg), detecta duplicatas por hash do cube e
gera um relatório. Opcionalmente remove duplicatas mantendo apenas um exemplar.

Uso:
  # Só relatar duplicatas e contagens
  python scripts/consolidate_windows.py --root .

  # Remover duplicatas (mantém o primeiro encontrado)
  python scripts/consolidate_windows.py --root . --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def hash_cube(path: Path) -> Tuple[str, Dict]:
    """Retorna SHA1 do cube e meta decodificada."""
    z = np.load(path, allow_pickle=True)
    cube = z["cube"]
    h = hashlib.sha1(cube.tobytes()).hexdigest()
    meta_raw = z.get("meta")
    meta: Dict = {}
    if meta_raw is not None:
        try:
            meta = json.loads(meta_raw.item()) if hasattr(meta_raw, "item") else json.loads(meta_raw)
        except Exception:
            meta = {}
    return h, meta


def main():
    ap = argparse.ArgumentParser(description="Detecta/remover duplicatas de janelas *.npz.")
    ap.add_argument("--root", type=str, default=".", help="Raiz para varrer (default=.)")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Se setado, remove duplicatas mantendo apenas o primeiro encontrado.",
    )
    args = ap.parse_args()
    root = Path(args.root).resolve()

    files = [p for p in root.rglob("*.npz") if "windows" in str(p).lower()]
    print(f"[SCAN] encontrados {len(files)} arquivos *.npz contendo 'windows' no caminho.")

    by_hash: Dict[str, List[Path]] = defaultdict(list)
    pos = neg = unknown = 0

    for p in sorted(files):
        try:
            h, meta = hash_cube(p)
            label = meta.get("label")
            if label is None:
                if "pos" in p.name.lower():
                    label = 1
                elif "neg" in p.name.lower():
                    label = 0
            if label == 1:
                pos += 1
            elif label == 0:
                neg += 1
            else:
                unknown += 1
            by_hash[h].append(p)
        except Exception as e:
            print(f"[WARN] falha lendo {p}: {e}")

    dups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}

    print(f"[COUNT] POS={pos} NEG={neg} UNKNOWN={unknown}")
    print(f"[HASH] únicos={len(by_hash)} duplicatas={len(dups)}")

    if dups:
        print("\n[LIST] grupos duplicados (hash -> caminhos):")
        for h, ps in list(dups.items())[:5]:
            print(f"  hash {h}:")
            for p in ps:
                print(f"    - {p}")
        if len(dups) > 5:
            print(f"  ... (+{len(dups)-5} grupos)")

    if args.apply and dups:
        removed = 0
        for h, ps in dups.items():
            keep = ps[0]
            drop = ps[1:]
            for p in drop:
                try:
                    os.remove(p)
                    removed += 1
                except Exception as e:
                    print(f"[WARN] não removi {p}: {e}")
            print(f"[CLEAN] hash {h}: mantido {keep.name}, removidos {len(drop)}")
        print(f"[DONE] removidos {removed} arquivos duplicados.")
    elif args.apply:
        print("[DONE] nada a remover (sem duplicatas).")


if __name__ == "__main__":
    main()
