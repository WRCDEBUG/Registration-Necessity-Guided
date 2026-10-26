#!/usr/bin/env python3
"""Cache task-importance maps derived from detection annotations.

For every image pair we rasterise the annotated boxes into a Gaussian-smoothed
importance map, downsampled to 120x160 and stored as uint8 .npy.  This map is
the S_task term of the measured necessity target.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from necessity_common import boxes_to_importance, load_yolo_boxes  # noqa: E402


def voc_to_yolo_boxes(xml_path: Path):
    if not xml_path.exists():
        return [], None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append(((xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H,
                      (xmax - xmin) / W, (ymax - ymin) / H))
    return boxes, (W, H)


def process_llvip(root: Path, ann: Path, out: Path, split: str, h: int, w: int,
                  sigma: float, yolo_out: Path | None):
    ir_dir = root / "infrared" / split
    out_dir = out / split
    out_dir.mkdir(parents=True, exist_ok=True)
    if yolo_out is not None:
        (yolo_out / split).mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in ir_dir.glob("*.jpg")) + \
        sorted(p.stem for p in ir_dir.glob("*.png"))
    n_empty = 0
    for i, name in enumerate(names):
        boxes, _ = voc_to_yolo_boxes(ann / f"{name}.xml")
        imp = boxes_to_importance(boxes, h, w, sigma=sigma)
        if imp.max() <= 0:
            n_empty += 1
        np.save(out_dir / f"{name}.npy",
                (imp * 255).astype(np.uint8))
        if yolo_out is not None:
            lines = []
            for cx, cy, bw, bh in boxes:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (yolo_out / split / f"{name}.txt").write_text("\n".join(lines))
        if (i + 1) % 1000 == 0:
            print(f"  LLVIP/{split}: {i + 1}/{len(names)}")
    print(f"[LLVIP/{split}] {len(names)} maps (empty: {n_empty})")


def process(root: Path, out: Path, split: str, h: int, w: int, sigma: float):
    ir_dir = root / "infrared" / split
    label_dir = root / "labels" / split
    out_dir = out / split
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in ir_dir.glob("*.png")) + \
        sorted(p.stem for p in ir_dir.glob("*.jpg"))
    n_empty = 0
    for i, name in enumerate(names):
        boxes = load_yolo_boxes(label_dir / f"{name}.txt")
        imp = boxes_to_importance(boxes, h, w, sigma=sigma)
        if imp.max() <= 0:
            n_empty += 1
        small = np.asarray(
            Image.fromarray((imp * 255).astype(np.uint8)).resize((w, h),
                                                                 Image.BILINEAR),
            dtype=np.uint8)
        np.save(out_dir / f"{name}.npy", small)
        if (i + 1) % 500 == 0:
            print(f"  {split}: {i + 1}/{len(names)}")
    print(f"[{root.name}/{split}] {len(names)} maps -> {out_dir} "
          f"(empty: {n_empty})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3fd", type=Path, default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--llvip", type=Path, default=Path("/home/ubuntu/Code/paper05/datasets/LLVIP"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                                 "revision_2026-09/cache/task_importance"))
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="Gaussian sigma in the low-resolution cache (px)")
    ap.add_argument("--llvip_ann", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/LLVIP/Annotations"))
    ap.add_argument("--llvip_yolo_out", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                                 "revision_2026-09/cache/llvip_yolo_labels"))
    args = ap.parse_args()

    for split in ("train", "test"):
        process(args.m3fd, args.out / "M3FD", split, args.h, args.w, args.sigma)
    if args.llvip.exists() and args.llvip_ann.exists():
        for split in ("train", "test"):
            process_llvip(args.llvip, args.llvip_ann, args.out / "LLVIP", split,
                          args.h, args.w, args.sigma, args.llvip_yolo_out)


if __name__ == "__main__":
    main()
