#!/usr/bin/env python3
"""E4: non-rigid / parallax-like diagnostic (reviewer R1.3).

The controlled protocol so far uses a global affine warp, so the per-pixel
displacement field is smooth and low-dimensional. Here we replace it with a
**layered parallax** warp: a small number of smooth depth-layer masks, each shifted
by its own translation (plus an optional global affine), which produces a
spatially varying, discontinuous-velocity field of the kind that parallax
creates. The forward mapping is analytic, so the ground-truth correction
displacement and the oracle registered candidate remain exact.

Conditions (same detector, same 300 ids as the magnitude sweep):
  none        : no correction
  full        : oracle registered candidate, N = 1
  geom        : N = normalise(d_gt)
  task        : N = predicted task map
  factorised  : N = normalise(d_gt * (phi + (1-phi) * task))
  uniform     : N = mean of the factorised map
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
from necessity_common import NecessityNet, build_target  # noqa: E402


def layer_masks(h: int, w: int, rng: np.random.Generator, n_layers: int,
                device) -> torch.Tensor:
    """Smooth depth-layer masks (n_layers, h, w) summing to one."""
    coarse = rng.normal(size=(n_layers, 4, 5)).astype(np.float32)
    t = torch.from_numpy(coarse)[None].to(device)
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)[0]
    m = torch.softmax(t * 2.0, dim=0)
    return m / m.sum(0, keepdim=True)


def parallax_flow(h: int, w: int, rng: np.random.Generator, amp_px: float,
                  n_layers: int, device):
    """Forward displacement u(x) with per-layer translations (pyramid up)."""
    masks = layer_masks(h, w, rng, n_layers, device)
    # depth-ordered translations: farther layers move less
    dirs = rng.normal(size=(n_layers, 2)).astype(np.float32)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6)
    scales = np.linspace(0.35, 1.0, n_layers).astype(np.float32)
    vec = torch.from_numpy(dirs * scales[:, None] * amp_px).to(device)   # (L,2)
    ux = (masks * vec[:, 0, None, None]).sum(0)
    uy = (masks * vec[:, 1, None, None]).sum(0)
    # add a small smooth non-affine residual (elastic component)
    r = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    rt = torch.from_numpy(r).to(device)
    rt = F.interpolate(rt, size=(h, w), mode="bicubic", align_corners=False)[0]
    rt = rt - rt.mean(dim=(1, 2), keepdim=True)
    n = torch.sqrt((rt ** 2).sum(0, keepdim=True)) + 1e-8
    rt = rt / n * (0.25 * amp_px)
    return ux + rt[0], uy + rt[1], masks


def warp_with_flow(img: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """ir_mis(x) = ir(x + u(x)) via grid_sample."""
    h, w = ux.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32),
                            torch.arange(w, device=img.device, dtype=torch.float32),
                            indexing="ij")
    sx, sy = xs + ux, ys + uy
    gx = 2.0 * sx / max(w - 1, 1) - 1.0
    gy = 2.0 * sy / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def invert_flow(ux: torch.Tensor, uy: torch.Tensor, iters: int = 4):
    """First-order + fixed-point inverse of x -> x + u(x): returns T^{-1}(x) - x."""
    h, w = ux.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=torch.device(ux.device), dtype=torch.float32),
                            torch.arange(w, device=torch.device(ux.device), dtype=torch.float32),
                            indexing="ij")
    # sample u at displaced coordinates (bilinear)
    def sample(field, px, py):
        gx = 2.0 * px / max(w - 1, 1) - 1.0
        gy = 2.0 * py / max(h - 1, 1) - 1.0
        return F.grid_sample(field[None, None], torch.stack([gx, gy], -1)[None],
                             mode="bilinear", padding_mode="border",
                             align_corners=True)[0, 0]
    vx = -ux
    vy = -uy
    for _ in range(iters):
        vx = -sample(ux, xs + vx, ys + vy)
        vy = -sample(uy, xs + vx, ys + vy)
    return vx, vy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--label_dir", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test"))
    ap.add_argument("--task_ckpt", type=Path,
                    default=REPO / "过程文件/revision_2026-09/results/task_importance_seed0/best.pt")
    ap.add_argument("--yolo", default=str(REPO / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--amp_px", type=float, default=8.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--output", type=Path,
                    default=REPO / "sensors-4626757/exp/nonrigid/run")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    task = NecessityNet().to(device).eval()
    ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
    task.load_state_dict(ck["model"] if "model" in ck else ck)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))[: args.max_samples]

    conds = ["none", "full", "geom", "task", "factorised", "uniform"]
    for c in conds:
        (args.output / c).mkdir(parents=True, exist_ok=True)

    stats = []
    with torch.no_grad():
        for idx, name in enumerate(names):
            ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                                args.img_h, args.img_w, device)
            ux, uy, _ = parallax_flow(args.img_h, args.img_w, rng, args.amp_px,
                                      args.layers, device)
            ir_mis = warp_with_flow(ir, ux, uy)
            vx, vy = invert_flow(ux, uy)
            d_gt = torch.sqrt(vx ** 2 + vy ** 2)
            ir_reg = warp_with_flow(ir_mis, vx, vy)     # oracle registered candidate
            stats.append(float(d_gt.mean()))

            logits = task(ir_mis, vi.mean(1, keepdim=True))
            s_hat = torch.sigmoid(F.interpolate(logits, size=(args.img_h, args.img_w),
                                               mode="bilinear", align_corners=False))
            g = build_target(d_gt[None, None], s_hat, args.geom_scale, args.floor)[0]
            g_geom = build_target(d_gt[None, None], torch.zeros_like(s_hat),
                                  args.geom_scale, 1.0)[0]
            gu = torch.full_like(g, float(g.mean()))

            ir3 = ir_mis.repeat(1, 3, 1, 1)
            f_mis = (0.5 * ir3 + 0.5 * vi).clamp(0, 1)
            f_reg = (0.5 * ir_reg.repeat(1, 3, 1, 1) + 0.5 * vi).clamp(0, 1)
            delta = f_reg - f_mis
            outs = {
                "none": f_mis,
                "full": f_reg,
                "geom": (f_mis + g_geom * delta).clamp(0, 1),
                "task": (f_mis + s_hat[0] * delta).clamp(0, 1),
                "factorised": (f_mis + g * delta).clamp(0, 1),
                "uniform": (f_mis + gu * delta).clamp(0, 1),
            }
            for c, t in outs.items():
                save_img(t, args.output / c / f"{name}.png")
            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(names)}")

    meta = {"n": len(names), "amp_px": args.amp_px, "layers": args.layers,
            "seed": args.seed, "mean_gt_displacement_px": float(np.mean(stats)),
            "median_gt_displacement_px": float(np.median(stats))}
    (args.output / "protocol.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    res = {}
    for c in conds:
        res[c] = run_yolo_eval(args.output / c, args.yolo, args.label_dir,
                               args.img_h, args.img_w)
        print(f"  {c}: mAP50={res[c]['mAP50']:.4f}")
    (args.output / "detection_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
