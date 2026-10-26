#!/usr/bin/env python3
"""E2b: simulated registration module with a *calibrated, spatially smooth* error field.

A deployed registration module returns a displacement field u_hat.  We model
u_hat = u_gt + e, where e is a smooth random field whose magnitude is set by
--err_px (mean |e| in pixels).  Both the warp and the geometric factor are
derived from the SAME u_hat, so the comparison is internally consistent:

  C4  : warp(u_hat), N = 1                      (dense correction with the estimate)
  C3  : warp(u_hat), N = gate(d_hat)            (the proposed gate, d_hat = |u_hat|)
  C5  : warp(u_hat), N = gate(d_hat) * 1[|e|<tau_r]   (ORACLE reliability; upper bound)

`none` and `full_gt` (oracle warp, N=1) are level-independent references.
The true error |e| is known only for the C5 oracle upper bound; the network never
sees it.

Output: images/ per condition + detection_results.json + err_stats.json
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
    misalign_ir, sample_affine_params,
)


def smooth_error(h: int, w: int, mag: float, rng: np.random.Generator,
                 gh: int = 4, gw: int = 5, device="cuda"):
    """Smooth displacement error with mean magnitude ~ mag pixels."""
    low = rng.normal(size=(1, 2, gh, gw)).astype(np.float32)
    t = torch.from_numpy(low).to(device)
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)[0]
    # remove mean translation so the field is pure local error
    t = t - t.mean(dim=(1, 2), keepdim=True)
    # unit-direction field
    n = torch.sqrt((t ** 2).sum(0, keepdim=True)) + 1e-8
    t = t / n
    # spatially varying amplitude: smooth positive field in ~[0.15, 3], mean 1
    amp = rng.normal(size=(1, 1, gh, gw)).astype(np.float32)
    amp = torch.from_numpy(amp).to(device)
    amp = F.interpolate(amp, size=(h, w), mode="bicubic", align_corners=False)[0]
    amp = torch.exp(1.2 * amp / (amp.std() + 1e-6))
    amp = amp.squeeze(0)
    amp = amp / amp.mean()
    e = t * amp * mag
    # renormalise so that mean |e| == mag
    m = torch.sqrt((e ** 2).sum(0)).mean() + 1e-8
    e = e * (mag / m)
    return e[0], e[1]  # ex, ey


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--err_px", type=float, default=1.5)
    ap.add_argument("--tau_r", type=float, default=2.0,
                    help="reliability threshold on |e| for the C5 oracle gate")
    ap.add_argument("--task_ckpt", type=Path,
                    default=REPO / "过程文件/revision_2026-09/results/task_importance_seed0/best.pt")
    ap.add_argument("--yolo", default=str(REPO / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--label_dir", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test"))
    ap.add_argument("--output", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/sim")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--skip_yolo", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    task_model = None
    if args.task_ckpt.exists():
        ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
        task_model = NecessityNet().to(device).eval()
        task_model.load_state_dict(ck["model"] if "model" in ck else ck)

    conds = ["none", "full_gt", "C3", "C4", "C5"]
    args.output.mkdir(parents=True, exist_ok=True)
    for c in conds:
        (args.output / "images" / c).mkdir(parents=True, exist_ok=True)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if args.max_samples:
        names = names[:args.max_samples]
    print(f"[sim] err_px={args.err_px} tau_r={args.tau_r} pairs={len(names)}")

    epes, rel_ratio = [], []
    for idx, name in enumerate(names):
        ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                            args.img_h, args.img_w, device)
        params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
        ir_mis = misalign_ir(ir, params)
        ux, uy, d_gt = displacement_field(args.img_h, args.img_w, *params, device=device)
        ex, ey = smooth_error(args.img_h, args.img_w, args.err_px, rng, device=device)
        hx, hy = ux + ex, uy + ey                      # the module's belief
        d_hat = torch.sqrt(hx ** 2 + hy ** 2)
        err = torch.sqrt(ex ** 2 + ey ** 2)
        epes.append(float(err.mean()))
        rel_ratio.append(float((err < args.tau_r).float().mean()))

        ys, xs = torch.meshgrid(
            torch.arange(args.img_h, device=device, dtype=torch.float32),
            torch.arange(args.img_w, device=device, dtype=torch.float32), indexing="ij")
        sx, sy = xs + hx, ys + hy
        gx = 2.0 * sx / max(args.img_w - 1, 1) - 1.0
        gy = 2.0 * sy / max(args.img_h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], -1)[None]
        ir_reg = F.grid_sample(ir_mis, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        # oracle GT warp (reference)
        _, Tinv = affine_matrices(args.img_h, args.img_w, *params)
        ys2, xs2 = torch.meshgrid(
            torch.arange(args.img_h, device=device, dtype=torch.float32),
            torch.arange(args.img_w, device=device, dtype=torch.float32), indexing="ij")
        P = torch.stack([xs2, ys2, torch.ones_like(xs2)], -1).reshape(-1, 3)
        Q = P @ Tinv.to(device).float()[:2, :3].T
        ggx = 2.0 * Q[:, 0].reshape(1, args.img_h, args.img_w) / (args.img_w - 1) - 1.0
        ggy = 2.0 * Q[:, 1].reshape(1, args.img_h, args.img_w) / (args.img_h - 1) - 1.0
        ir_reg_gt = F.grid_sample(ir_mis, torch.stack([ggx, ggy], -1),
                                  mode="bilinear", padding_mode="border",
                                  align_corners=True)

        if task_model is not None:
            logits = task_model(ir_mis, vi.mean(1, keepdim=True))
            s_hat = torch.sigmoid(F.interpolate(
                logits, size=(args.img_h, args.img_w), mode="bilinear",
                align_corners=False))
        else:
            s_hat = torch.zeros(1, 1, args.img_h, args.img_w, device=device)

        g = build_target(d_hat[None, None], s_hat, args.geom_scale, args.floor)[0]
        rel_mask = (err < args.tau_r).float()[None]
        g_rel = g * rel_mask

        ir3 = ir_mis.repeat(1, 3, 1, 1)
        f_mis = (0.5 * ir3 + 0.5 * vi).clamp(0, 1)
        f_gt = (0.5 * ir_reg_gt.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        f_est = (0.5 * ir_reg.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
        d_est = f_est - f_mis
        outs = {
            "none": f_mis,
            "full_gt": f_gt,
            "C3": f_mis + args.gamma * g * d_est,
            "C4": f_mis + args.gamma * d_est,
            "C5": f_mis + args.gamma * g_rel * d_est,
        }
        for c, t in outs.items():
            save_img(t.clamp(0, 1), args.output / "images" / c / f"{name}.png")
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(names)}")

    stats = {"err_px_target": args.err_px, "epe_mean": float(np.mean(epes)),
             "epe_median": float(np.median(epes)),
             "reliable_fraction": float(np.mean(rel_ratio)), "tau_r": args.tau_r}
    (args.output / "err_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

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
