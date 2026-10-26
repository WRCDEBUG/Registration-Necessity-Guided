#!/usr/bin/env python3
"""Per-image detection analysis, per-class metrics, and paired bootstrap.

For every condition (image directory) we run the frozen detector image by
image, match predictions to ground truth at IoU 0.5, and store:
  * per-image AP50 (area under that image's precision-recall curve),
  * TP / FP / FN counts, per class,
  * localisation quality (mean IoU of matched boxes).

The per-image AP vectors are then used for a paired bootstrap between
conditions, which yields a confidence interval for the difference in mAP50
without re-running the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def load_gt(label_path: Path, W: int, H: int):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            boxes.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H))
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def per_image_ap50(dets, gts, iou_thr=0.5):
    """Image-level AP50 (all classes pooled, single-image PR curve)."""
    if not gts:
        return float("nan")
    if not dets:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[1])
    matched = [False] * len(gts)
    tp, fp = [], []
    for cls, conf, box in dets:
        best, best_i = 0.0, -1
        for i, g in enumerate(gts):
            if matched[i] or g[0] != cls:
                continue
            v = iou(box, g[1:])
            if v > best:
                best, best_i = v, i
        if best >= iou_thr:
            matched[best_i] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / len(gts)
    prec = tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap += p / 101
    return float(ap)


@torch.no_grad()
def analyse_condition(img_dir: Path, label_dir: Path, model, device,
                      conf=0.25, iou_thr=0.5):
    rows = []
    names = sorted(p.stem for p in img_dir.glob("*.png"))
    for name in names:
        img_path = img_dir / f"{name}.png"
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        res = model.predict(np.asarray(img), imgsz=max(H, W), conf=conf,
                            verbose=False, device=device)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, confs, clss):
                dets.append((int(k), float(c), tuple(float(v) for v in b)))
        gts = load_gt(label_dir / f"{name}.txt", W, H)
        ap50 = per_image_ap50(dets, gts, iou_thr)

        matched_gt = set()
        tp = fp = 0
        ious = []
        for cls, confv, box in sorted(dets, key=lambda d: -d[1]):
            best, best_i = 0.0, -1
            for i, g in enumerate(gts):
                if i in matched_gt or g[0] != cls:
                    continue
                v = iou(box, g[1:])
                if v > best:
                    best, best_i = v, i
            if best >= iou_thr:
                matched_gt.add(best_i); tp += 1; ious.append(best)
            else:
                fp += 1
        fn = len(gts) - len(matched_gt)
        rows.append(dict(name=name, ap50=ap50, tp=tp, fp=fp, fn=fn,
                         mean_iou=float(np.mean(ious)) if ious else float("nan"),
                         n_gt=len(gts)))
    return rows


def bootstrap_diff(a, b, n=5000, seed=0):
    """Paired bootstrap of mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return dict(mean_diff=float(d.mean()),
                ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)),
                p_gt0=float((boots > 0).mean()), n=int(len(d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=Path, required=True)
    ap.add_argument("--label_dir", type=Path, required=True)
    ap.add_argument("--yolo", type=str,
                    default=str(ROOT / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--conditions", type=str, default="")
    ap.add_argument("--baseline", type=str, default="none")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = YOLO(args.yolo)
    except Exception:
        model = RTDETR(args.yolo)
    args.output.mkdir(parents=True, exist_ok=True)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()] or \
        sorted(p.name for p in (args.images_root).iterdir() if p.is_dir())

    per_image = {}
    for c in conds:
        d = args.images_root / c
        if not d.exists():
            continue
        rows = analyse_condition(d, args.label_dir, model, device, args.conf)
        if args.max_samples:
            rows = rows[:args.max_samples]
        per_image[c] = rows
        with (args.output / f"per_image_{c}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"{c:24s} mAP50(per-image mean)={np.nanmean([r['ap50'] for r in rows]):.4f} "
              f"TP={sum(r['tp'] for r in rows)} FP={sum(r['fp'] for r in rows)} "
              f"FN={sum(r['fn'] for r in rows)}")

    summary = {}
    if args.baseline in per_image:
        base = [r["ap50"] for r in per_image[args.baseline]]
        for c, rows in per_image.items():
            if c == args.baseline:
                continue
            summary[c] = bootstrap_diff([r["ap50"] for r in rows], base)
    (args.output / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
