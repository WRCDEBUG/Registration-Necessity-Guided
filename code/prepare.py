#!/usr/bin/env python3
"""Build YOLO training trees for the detector-bias check (reviewer round 2, item 4).

Creates a train/val split *inside* the M3FD training pool only (the 1049-pair test
split is never used for training or checkpoint selection), and lays out three image
sources:

  sage      SAGE-fused training pairs        (different fusion source)
  visible   raw visible images               (no fusion at all)

Labels are shared (M3FD YOLO boxes).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def build(images_dir: Path, labels_dir: Path, out: Path, names: list[str],
          val_every: int = 9) -> int:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    n_val = 0
    for i, p in enumerate(files):
        split = "val" if (i % val_every == 0) else "train"
        n_val += split == "val"
        link(p, out / "images" / split / p.name)
        lab = labels_dir / f"{p.stem}.txt"
        if lab.exists():
            link(lab, out / "labels" / split / lab.name)
        else:
            (out / "labels" / split / f"{p.stem}.txt").write_text("")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")
    print(f"{out.name}: {len(files)} images ({len(files)-n_val} train / {n_val} val)")
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/sensors-4626757/exp/detector_bias"))
    ap.add_argument("--sage", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/revision_2026-09/baselines/sage_train/SAGE-IVIF/images"))
    ap.add_argument("--visible", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/visible/train"))
    ap.add_argument("--labels", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/train"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.sage.exists():
        build(args.sage, args.labels, args.out / "sage", NAMES)
    else:
        print(f"[skip] SAGE images not ready: {args.sage}")
    build(args.visible, args.labels, args.out / "visible", NAMES)


if __name__ == "__main__":
    main()
