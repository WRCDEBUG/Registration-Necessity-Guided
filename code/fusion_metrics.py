#!/usr/bin/env python3
"""Fusion-quality metrics for the bounded-correction conditions (R1-M6).

Metrics: EN, SD, SF, MI, Qabf, VIF, SSIM(IR), SSIM(VIS), SCD.

References are the *observed* inputs of each fusion: the misaligned infrared image
(same seed rule as the detector evaluation) and the visible image. For the
`full` condition the fused image is built from the oracle-registered IR, so its
reference-based scores are conservative by construction.

The MI/Qabf/VIF/SSIM implementations are reused verbatim from the VF-Bench metric
module (tested, standard formulations); EN/SD/SF/SCD are computed here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
VFB = Path("/home/ubuntu/Code/VideoFusion/VF-Bench-main/src")
for p in (str(SCRIPTS), str(REPO), str(VFB)):
    sys.path.insert(0, p)

from util.metric import Metric_MI, Metric_Qabf, Metric_SSIM, Metric_VIF  # noqa: E402
from necessity_common import misalign_ir, sample_affine_params  # noqa: E402


def load_rgb(path: Path, h: int, w: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def entropy(g: np.ndarray) -> float:
    hist, _ = np.histogram(g, bins=256, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def spatial_freq(g: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(g, axis=0) ** 2)) if g.shape[0] > 1 else 0.0
    cf = np.sqrt(np.mean(np.diff(g, axis=1) ** 2)) if g.shape[1] > 1 else 0.0
    return float(np.sqrt(rf ** 2 + cf ** 2))


def scd(f: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Sum of correlations of differences (Aslantas & Bendes)."""
    def corr(x, y):
        x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
        return float((x * y).sum() / d)
    return corr(f - b, a - b) + corr(f - a, b - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=REPO / "sensors-4626757/exp/gates/e1_seed0/images")
    ap.add_argument("--geom_root", type=Path,
                    default=REPO / "sensors-4626757/exp/gates/geom_rules/images")
    ap.add_argument("--conditions", default="none,full,uniform_g1,learned_factorized_g1,task_only_g1,geom_thresh_g1")
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--ref_mode", choices=["mis", "clean"], default="mis",
                    help="reference infrared: the observed (misaligned) IR or the clean IR")
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/fusion/fusion_quality.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))[: args.max_samples]

    rows = []
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        d = args.root / cond
        if not d.exists():
            d = args.geom_root / cond
        if not d.exists():
            print(f"[q] missing condition dir: {cond}")
            continue
        acc = {k: [] for k in ("EN", "SD", "SF", "MI", "Qabf", "VIF", "SSIM", "SCD")}
        n = 0
        for idx, name in enumerate(names):
            fp = d / f"{name}.png"
            if not fp.exists():
                continue
            fused = load_rgb(fp, args.img_h, args.img_w)
            vi = load_rgb(vi_dir / f"{name}.png", args.img_h, args.img_w)
            ir_clean = torch.from_numpy(np.asarray(
                Image.open(ir_dir / f"{name}.png").convert("L").resize(
                    (args.img_w, args.img_h), Image.BILINEAR), dtype=np.float32) / 255.0)[None, None]
            params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
            ir_mis = misalign_ir(ir_clean, params)
            ref_src = ir_mis if args.ref_mode == "mis" else ir_clean
            ref_ir = ref_src.repeat(1, 3, 1, 1).to(device)
            ref_vi = vi[None].to(device)
            f5 = fused[None].to(device)
            g = f5.mean(1)[0].detach().cpu().numpy()
            gi = ref_ir.mean(1)[0].detach().cpu().numpy()
            gv = ref_vi.mean(1)[0].detach().cpu().numpy()

            acc["EN"].append(entropy(g))
            acc["SD"].append(float(g.std()))
            acc["SF"].append(spatial_freq(g))
            acc["SCD"].append(scd(g, gi, gv))
            with torch.no_grad():
                acc["MI"].append(float(Metric_MI(f5 * 255.0, ref_ir * 255.0, ref_vi * 255.0).mean()))
                acc["Qabf"].append(float(Metric_Qabf(f5, ref_ir, ref_vi).mean()))
                acc["VIF"].append(float(Metric_VIF(f5, ref_ir, ref_vi).mean()))
                acc["SSIM"].append(float(Metric_SSIM(f5, ref_ir, ref_vi).mean()))
            n += 1
        row = {"condition": cond, "n": n, "ref": args.ref_mode}
        row.update({k: f"{np.mean(v):.4f}" for k, v in acc.items()})
        rows.append(row)
        print(f"[q] {cond}: n={n} " + " ".join(f"{k}={row[k]}" for k in ("EN", "SF", "MI", "Qabf", "VIF")))

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md = args.out.with_suffix(".md")
    with md.open("w") as f:
        f.write(f"# Fusion-quality metrics (pixel-average family, infrared reference = {args.ref_mode})\n\n")
        f.write("| condition | ref | n | EN | SD | SF | MI | Qabf | VIF | SSIM | SCD |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| {condition} | {ref} | {n} | {EN} | {SD} | {SF} | {MI} | {Qabf} | {VIF} | {SSIM} | {SCD} |\n".format(**r))
        f.write("\nReferences are the observed (misaligned) infrared image and the visible image; "
                "the `full` condition is built from the oracle-registered IR, so its MI/Qabf/VIF/SSIM "
                "are conservative.\n")
    print(f"[q] wrote {args.out} and {md}")


if __name__ == "__main__":
    main()
