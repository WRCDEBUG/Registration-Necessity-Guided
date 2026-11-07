#!/usr/bin/env python3
"""Mixed-magnitude evaluation set (reviewer round 2, item 2).

The ±2 px evidence came from a 300-pair subset, and the full test set has no
small-misalignment subgroup (10th percentile of mean displacement 5.85 px). This
script rebuilds the evaluation set with a *controlled mixture* of magnitudes on all
1049 test pairs, so that a per-image "correct or not" decision can be tested against
always-correct on the same images.

Magnitude strata (deterministic, by sorted index):
  first 30 %  :  1-3 px   (below the 4 px threshold the reviewer asks for)
  next  35 %  :  4-10 px
  last  35 %  : 12-24 px
Rotation and scale are scaled proportionally to the translation so that the mean
displacement follows the target magnitude.

Outputs none/ and full/ image dirs plus meta.csv (name, target, mean_d).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件/revision_2026-09/scripts"
GATES = REPO / "sensors-4626757/exp/gates"
for p in (str(SCRIPTS), str(GATES)):
    sys.path.insert(0, p)

from eval_gates_ext import _load_pair, save_img  # noqa: E402
from necessity_common import affine_matrices, displacement_field, misalign_ir, warp_by_matrix  # noqa: E402


def target_magnitude(i: int, n: int, rng: np.random.Generator) -> float:
    frac = i / max(n - 1, 1)
    if frac < 0.30:
        return float(rng.uniform(1.5, 3.8))
    if frac < 0.65:
        return float(rng.uniform(4.0, 10.0))
    return float(rng.uniform(12.0, 24.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/mixed/eval")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    for c in ("none", "full"):
        (args.out / c).mkdir(parents=True, exist_ok=True)

    rows = []
    for i, name in enumerate(names):
        mag = target_magnitude(i, len(names), rng)
        angle = 3.0 * mag / 8.0 * rng.choice([-1, 1])
        sign = rng.choice([-1, 1], size=2)
        translate = tuple(float(s * mag) for s in sign)
        scale = 1.0 + 0.03 * mag / 8.0 * rng.choice([-1, 1])
        ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                            args.img_h, args.img_w, device)
        # rescale the provisional transform so the measured mean displacement matches
        # the target magnitude (one correction step is enough in practice)
        d0 = displacement_field(args.img_h, args.img_w, angle, translate, scale, device=device)[2]
        k = mag / max(float(d0.mean()), 1e-6)
        angle *= k
        translate = (translate[0] * k, translate[1] * k)
        ir_mis = misalign_ir(ir, (angle, translate, scale))
        _, Tinv = affine_matrices(args.img_h, args.img_w, angle, translate, scale)
        ir_reg = warp_by_matrix(ir_mis, Tinv.to(device).float())
        f_mis = (0.5 * ir_mis.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        f_reg = (0.5 * ir_reg.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        save_img(f_mis, args.out / "none" / f"{name}.png")
        save_img(f_reg, args.out / "full" / f"{name}.png")
        d = displacement_field(args.img_h, args.img_w, angle, translate, scale, device=device)[2]
        rows.append(dict(name=name, target_px=f"{mag:.3f}", mean_d=f"{float(d.mean()):.3f}",
                         angle=f"{angle:.3f}", tx=f"{translate[0]:.3f}", ty=f"{translate[1]:.3f}",
                         scale=f"{scale:.5f}"))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(names)}")

    with (args.out / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    m = np.array([float(r["mean_d"]) for r in rows])
    print(f"[mixed] {len(rows)} pairs, mean displacement: "
          f"min {m.min():.2f}  p10 {np.percentile(m,10):.2f}  median {np.median(m):.2f}  max {m.max():.2f}")
    print(f"[mixed] share below 4 px: {(m < 4).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
