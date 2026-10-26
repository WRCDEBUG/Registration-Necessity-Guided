#!/usr/bin/env python3
"""Image-level necessity gate: decide *per image* whether to correct at all.

Motivation (E1/E2 evidence): pixel-level gating never beats dense correction, but
(i) full correction is worse than no correction at very small misalignment and
(ii) per-image harm exists, so the decision may belong at the image level.

For each test image we take the already-generated `none` and `full` fused images
(no new fusion) and select one of them according to a per-image necessity score:

  oracle : mean ground-truth displacement      d_bar_img
  est    : mean displacement from the real estimated registration matrix

Then a threshold tau selects "correct" (full) iff score >= tau.  This is a
single scalar per image, so the experiment is cheap and uses the same detector.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
for p in (str(SCRIPTS), str(REPO)):
    sys.path.insert(0, p)

from eval.eval_tdsr import run_yolo_eval  # noqa: E402
from necessity_common import displacement_field, sample_affine_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=REPO / "sensors-4626757/exp/gates/e1_seed0/images")
    ap.add_argument("--est", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/est_v2_test.npz")
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--label_dir", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test"))
    ap.add_argument("--yolo", default=str(REPO / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--taus", default="2,3,4,5,6,8")
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/gates/image_level")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ir_dir = args.data_root / "infrared" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))

    est = np.load(args.est)
    est_idx = {str(n): i for i, n in enumerate(est["names"])}
    W = est["W"]

    scores = {"oracle_mean": {}, "est_mean": {}}
    for idx, name in enumerate(names):
        params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
        _, _, mag = displacement_field(480, 640, *params, device=device)
        scores["oracle_mean"][name] = float(mag.mean())
        if name in est_idx:
            w = W[est_idx[name]].reshape(2, 3)
            ys, xs = np.mgrid[0:480, 0:640].astype(np.float64)
            ux = w[0, 0] * xs + w[0, 1] * ys + w[0, 2] - xs
            uy = w[1, 0] * xs + w[1, 1] * ys + w[1, 2] - ys
            scores["est_mean"][name] = float(np.sqrt(ux ** 2 + uy ** 2).mean())

    for kind in ("oracle_mean", "est_mean"):
        v = np.array([scores[kind][n] for n in names if n in scores[kind]])
        print(f"[img] {kind}: n={len(v)} mean={v.mean():.2f} min={v.min():.2f} "
              f"p25={np.percentile(v,25):.2f} median={np.median(v):.2f} p75={np.percentile(v,75):.2f} max={v.max():.2f}")

    results = {}
    for kind in ("oracle_mean", "est_mean"):
        for tau in [float(t) for t in args.taus.split(",")]:
            tag = f"{kind}_tau{tau:g}"
            d = args.out / tag
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            n_corr = 0
            for idx, name in enumerate(names):
                sc = scores[kind].get(name)
                if sc is None:
                    continue
                src = args.root / ("full" if sc >= tau else "none") / f"{name}.png"
                if sc >= tau:
                    n_corr += 1
                (d / f"{name}.png").symlink_to(src)
            res = run_yolo_eval(d, args.yolo, args.label_dir, args.img_h, args.img_w)
            res["n_corrected"] = n_corr
            results[tag] = res
            print(f"[img] {tag}: mAP50={res['mAP50']:.4f} corrected={n_corr}/{len(names)}")

    (args.out / "image_level_results.json").write_text(json.dumps(results, indent=2))
    (args.out / "scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
