#!/usr/bin/env python3
"""E2: evaluation with a REAL estimated registration.

Conditions (all on the same 1049 pairs, same detector, same task map):
  none        : no correction
  full_gt     : oracle warp, N = 1                (upper reference)
  C0          : oracle warp, N = gate(d_gt)       (paper's factorised gate, reproduced)
  C1          : oracle warp, N = gate(d_est)      (only the GEOMETRIC FACTOR is wrong)
  C2          : estimated warp, N = gate(d_gt)    (only the WARP is wrong)
  C3          : estimated warp, N = gate(d_est)   (full deployment: both from the estimate)
  C4          : estimated warp, N = 1             (dense correction with the estimate)

gate(d) = normalize( (d/s_g).clamp(0,1) * (phi + (1-phi) * S_hat) )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
GATES = REPO / "sensors-4626757" / "exp" / "gates"
for p in (str(SCRIPTS), str(GATES), str(REPO)):
    sys.path.insert(0, p)

from eval_gates_ext import _load_pair, save_img  # noqa: E402
from eval.eval_tdsr import run_yolo_eval  # noqa: E402
from necessity_common import (  # noqa: E402
    NecessityNet, affine_matrices, build_target, displacement_field,
    misalign_ir, sample_affine_params, warp_by_matrix,
)


def gate_from_d(d: torch.Tensor, s_hat: torch.Tensor, s_g: float, phi: float) -> torch.Tensor:
    return build_target(d[None, None], s_hat[None], s_g, phi)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--est", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/est_v2.npz")
    ap.add_argument("--task_ckpt", type=Path,
                    default=REPO / "过程文件/revision_2026-09/results/task_importance_seed0/best.pt")
    ap.add_argument("--yolo", default=str(REPO / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--label_dir", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test"))
    ap.add_argument("--output", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/eval")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--skip_yolo", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    est = np.load(args.est)
    est_idx = {str(n): i for i, n in enumerate(est["names"])}
    W_est = est["W"]

    task_model = None
    if args.task_ckpt.exists():
        ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
        task_model = NecessityNet().to(device).eval()
        task_model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"[e2] task model: {args.task_ckpt}")

    conds = ["none", "full_gt", "C0", "C1", "C2", "C3", "C4"]
    args.output.mkdir(parents=True, exist_ok=True)
    for c in conds:
        (args.output / "images" / c).mkdir(parents=True, exist_ok=True)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if args.max_samples:
        names = names[:args.max_samples]
    print(f"[e2] {len(names)} pairs, conditions={conds}")

    for idx, name in enumerate(names):
        if name not in est_idx:
            print(f"[e2] WARNING no estimate for {name}; skip")
            continue
        ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                            args.img_h, args.img_w, device)
        params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
        ir_mis = misalign_ir(ir, params)
        _, Tinv = affine_matrices(args.img_h, args.img_w, *params)
        d_gt = displacement_field(args.img_h, args.img_w, *params, device=device)[2]

        W = torch.from_numpy(W_est[est_idx[name]].reshape(2, 3)).float().to(device)
        ir_reg_gt = warp_by_matrix(ir_mis, Tinv.to(device).float())
        ir_reg_est = warp_by_matrix(ir_mis, W)
        # displacement implied by the estimated matrix
        ys, xs = torch.meshgrid(
            torch.arange(args.img_h, device=device, dtype=torch.float32),
            torch.arange(args.img_w, device=device, dtype=torch.float32), indexing="ij")
        ux = W[0, 0] * xs + W[0, 1] * ys + W[0, 2] - xs
        uy = W[1, 0] * xs + W[1, 1] * ys + W[1, 2] - ys
        d_est = torch.sqrt(ux ** 2 + uy ** 2)

        s_hat = None
        if task_model is not None:
            logits = task_model(ir_mis, vi.mean(1, keepdim=True))
            s_hat = torch.sigmoid(F.interpolate(
                logits, size=(args.img_h, args.img_w), mode="bilinear",
                align_corners=False))
        else:
            s_hat = torch.zeros(1, 1, args.img_h, args.img_w, device=device)

        g_gt = gate_from_d(d_gt, s_hat, args.geom_scale, args.floor)
        g_est = gate_from_d(d_est, s_hat, args.geom_scale, args.floor)
        ones = torch.ones_like(g_gt)

        ir_rgb_mis = ir_mis.repeat(1, 3, 1, 1)
        f_mis = (0.5 * ir_rgb_mis + 0.5 * vi).clamp(0, 1)
        f_gt = (0.5 * ir_reg_gt.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        f_est = (0.5 * ir_reg_est.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        d_gt_delta = f_gt - f_mis
        d_est_delta = f_est - f_mis

        outs = {
            "none": f_mis,
            "full_gt": f_gt,
            "C0": f_mis + args.gamma * g_gt * d_gt_delta,
            "C1": f_mis + args.gamma * g_est * d_gt_delta,
            "C2": f_mis + args.gamma * g_gt * d_est_delta,
            "C3": f_mis + args.gamma * g_est * d_est_delta,
            "C4": f_mis + args.gamma * ones * d_est_delta,
        }
        for c, t in outs.items():
            save_img(t.clamp(0, 1), args.output / "images" / c / f"{name}.png")
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(names)}")

    if not args.skip_yolo:
        res = {}
        for c in conds:
            print(f"  YOLO: {c}")
            res[c] = run_yolo_eval(args.output / "images" / c, args.yolo,
                                   args.label_dir, args.img_h, args.img_w)
        (args.output / "detection_results.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
