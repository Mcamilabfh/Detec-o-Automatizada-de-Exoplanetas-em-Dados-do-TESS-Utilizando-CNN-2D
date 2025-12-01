#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_masks.py
Relatório simples de máscaras before/during nos NPZ.
"""
from pathlib import Path
import numpy as np


def main():
    root = Path("data")
    files = [p for p in root.rglob("*.npz") if "windows" in str(p).lower()]
    total = len(files)
    with_masks = 0
    masks_empty = 0
    for p in files:
        z = np.load(p, allow_pickle=True)
        mb = z.get("mask_before")
        md = z.get("mask_during")
        if mb is None or md is None:
            continue
        with_masks += 1
        if md.sum() < 1 or mb.sum() < 1:
            masks_empty += 1
    print(f"Total NPZ: {total}")
    print(f"Com masks: {with_masks}")
    print(f"Masks vazias (md==0 ou mb==0): {masks_empty}")


if __name__ == "__main__":
    main()
