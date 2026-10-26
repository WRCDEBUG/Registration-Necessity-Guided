#!/usr/bin/env python3
"""Bounded correction on top of a learned fusion backbone (R1.6).

Given the fused outputs of a learned method on (a) the misaligned pair and
(b) the oracle-registered pair, we form the same bounded correction as in the
paper:

    F = F_mis + gamma * N * (F_reg - F_mis)

with N the factorised necessity map predicted from the image pair. Conditions:
none (=F_mis), full (=F_reg), gated (N), uniform (matched mean activation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
GATES = REPO / "sensors-4626757" / "exp" / "gates"
for p in (str(SCRIPTS), str(GATES), str(REPO)):
    sys.path.insert(0, p)

from eval_gates_ext import _load_pair, save_img  # noqa: E402
from eval.eval_tdsr import run_yolo_eval  # noqa: E402
from necessity_common import (  # noqa: E402
    NecessityNet, build_target, displacement_field, misalign_ir, sample_affine_params,
)


def load_rgb(path: Path, h: int, w: int, device) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg_dir", type=Path,
                    default=REPO / "sensors-4626757/exp/fusion/seafusion_aligned/SeAFusion/images")
    ap.add_argument("--mis_dir", type=Path,
                    default=REPO / "sensors-4626757/exp/fusion/seafusion_misaligned/SeAFusion/images")
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
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--output", type=Path,
                    default=REPO / "sensors-4626757/exp/fusion/gated_seafusion")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    task = NecessityNet().to(device).eval()
    ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
    task.load_state_dict(ck["model"] if "model" in ck else ck)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))[: args.max_samples]

    conds = ["none", "full", "gated", "uniform"]
    for c in conds:
        (args.output / c).mkdir(parents=True, exist_ok=True)

    used = 0
    with torch.no_grad():
        for idx, name in enumerate(names):
            fr = args.reg_dir / f"{name}.png"
            fm = args.mis_dir / f"{name}.png"
            if not (fr.exists() and fm.exists()):
                continue
            f_reg = load_rgb(fr, args.img_h, args.img_w, device)
            f_mis = load_rgb(fm, args.img_h, args.img_w, device)
            ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                                args.img_h, args.img_w, device)
            params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
            ir_mis = misalign_ir(ir, params)
            _, _, mag = displacement_field(args.img_h, args.img_w, *params, device=device)
            logits = task(ir_mis, vi.mean(1, keepdim=True))
            s_hat = torch.sigmoid(F.interpolate(logits, size=(args.img_h, args.img_w),
                                               mode="bilinear", align_corners=False))
            g = build_target(mag[None, None], s_hat, args.geom_scale, args.floor)[0]
            gu = torch.full_like(g, float(g.mean()))
            delta = f_reg - f_mis
            outs = {
                "none": f_mis,
                "full": f_reg,
                "gated": (f_mis + args.gamma * g * delta).clamp(0, 1),
                "uniform": (f_mis + args.gamma * gu * delta).clamp(0, 1),
            }
            for c, t in outs.items():
                save_img(t, args.output / c / f"{name}.png")
            used += 1
            if used % 100 == 0:
                print(f"  {used}/{len(names)}")

    print(f"[gated-learned] {used} pairs written")
    res = {}
    for c in conds:
        res[c] = run_yolo_eval(args.output / c, args.yolo, args.label_dir,
                               args.img_h, args.img_w)
        print(f"  {c}: mAP50={res[c]['mAP50']:.4f}")
    (args.output / "detection_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
