#!/usr/bin/env python3
"""Per-image AP50 for the VTMOT conditions + paired bootstrap (none vs H, H vs gate)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/sensors-4626757")


def load_gt(p: Path, w: int, h: int) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 5))
    rows = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 5))
    arr = np.array([[float(x) for x in r] for r in rows], np.float32).reshape(-1, 5)
    out = np.zeros_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1] = (arr[:, 1] - arr[:, 3] / 2) * w
    out[:, 2] = (arr[:, 2] - arr[:, 4] / 2) * h
    out[:, 3] = (arr[:, 1] + arr[:, 3] / 2) * w
    out[:, 4] = (arr[:, 2] + arr[:, 4] / 2) * h
    return out


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(dets, gt, iou_thr=0.5) -> float:
    if len(gt) == 0:
        return float("nan")
    used = np.zeros(len(gt), bool)
    order = np.argsort(-dets[:, 4]) if len(dets) else []
    tp, fp = [], []
    for i in order:
        d = dets[i]
        best, bi = 0.0, -1
        for j, g in enumerate(gt):
            if used[j] or int(g[0]) != int(d[0]):
                continue
            v = iou(d[1:5], g[1:5])
            if v > best:
                best, bi = v, j
        if best >= iou_thr and bi >= 0:
            used[bi] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not tp:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / len(gt)
    prec = tp / np.maximum(tp + fp, 1e-9)
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0.0
                          for r in np.linspace(0, 1, 11)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=BASE / "exp/vtmot/run_coco")
    ap.add_argument("--yolo", default=str(BASE.parent / "eval/yolov8s.pt"))
    ap.add_argument("--cond", nargs="+", default=["none", "H", "gate", "uniform"])
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    model = YOLO(args.yolo)
    per = {}
    for c in args.cond:
        rows = []
        for img in sorted((args.root / c).glob("*.png")):
            res = model.predict(str(img), imgsz=640, conf=args.conf, iou=args.iou,
                                classes=args.classes, verbose=False)[0]
            dets = np.zeros((0, 5))
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                dets = np.concatenate([cls[:, None], xyxy, conf[:, None]], 1)
                # label space: COCO person 0, car 2 -> GT 0 (People) / 2 (Car)
            probe = model.predict(str(img), imgsz=640, conf=args.conf, iou=args.iou,
                                  classes=args.classes, verbose=False)[0]
            hh, ww = probe.orig_shape
            gt = load_gt(args.root / "labels" / f"{img.stem}.txt", ww, hh)
            rows.append((img.stem, ap50(dets, gt)))
        per[c] = {n: v for n, v in rows if not np.isnan(v)}
        vals = np.array([v for n, v in rows if not np.isnan(v)])
        print(f"  {c}: n={len(vals)} per-image AP50 mean {vals.mean():.4f}")
        with (args.root / f"per_image_{c}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "ap50"])
            for n, v in rows:
                if not np.isnan(v):
                    w.writerow([n, f"{v:.6f}"])

    rng = np.random.default_rng(123)
    out = {}
    for a, b in (("none", "H"), ("H", "gate"), ("none", "gate"), ("gate", "uniform")):
        ids = sorted(set(per[a]) & set(per[b]))
        d = np.array([per[a][i] - per[b][i] for i in ids])
        boots = [float(np.mean(rng.choice(d, len(d), True))) for _ in range(5000)]
        out[f"{a}_minus_{b}"] = {"mean": float(d.mean()),
                                 "ci_low": float(np.percentile(boots, 2.5)),
                                 "ci_high": float(np.percentile(boots, 97.5)),
                                 "n": len(ids)}
        print(f"  {a} - {b}: {d.mean():+.4f} [{np.percentile(boots,2.5):+.4f},{np.percentile(boots,97.5):+.4f}] n={len(ids)}")
    (args.root / "bootstrap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
