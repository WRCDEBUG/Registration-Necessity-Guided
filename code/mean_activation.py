#!/usr/bin/env python3
"""Mean gate activation for every gate map, on the 1049 M3FD test pairs.

Purpose: test the hypothesis that downstream mAP50 is (mainly) a function of how
much correction is applied, i.e. the mean gate activation, rather than of the
spatial arrangement of the gate.

No images are written; only the gates are built (same code path as the evaluator).
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

from eval_gates_ext import _load_pair  # noqa: E402
from exp_a_necessity_validity import load_model as load_legacy_model  # noqa: E402
from necessity_common import (  # noqa: E402
    NecessityNet, boxes_to_importance, build_target, displacement_field,
    misalign_ir, sample_affine_params,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imp_root", type=Path,
                    default=REPO / "过程文件/revision_2026-09/cache/task_importance/M3FD")
    ap.add_argument("--task_ckpt", type=Path,
                    default=REPO / "过程文件/revision_2026-09/results/task_importance_seed0/best.pt")
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "过程文件/revision_2026-09/results/necessity_seed0/best.pt")
    ap.add_argument("--legacy_ckpt", type=Path,
                    default=REPO / "sensors-4626757/audit/recovered/tsdr_r1/"
                                      "results_full_proxy_mult_beta05_v2/checkpoints/"
                                      "stage3_best_spearman.pth")
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/gates/mean_activation.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = NecessityNet().to(device).eval()
    ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
    task.load_state_dict(ck["model"] if "model" in ck else ck)
    nec = NecessityNet().to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    nec.load_state_dict(ck["model"] if "model" in ck else ck)
    legacy = load_legacy_model(args.legacy_ckpt, device)
    legacy.gate_mode = "soft"

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    acc = {}

    with torch.no_grad():
        for idx, name in enumerate(names):
            ir, vi = _load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                                480, 640, device)
            params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
            ir_mis = misalign_ir(ir, params)
            _, _, mag = displacement_field(480, 640, *params, device=device)
            imp = torch.from_numpy(np.load(args.imp_root / args.split / f"{name}.npy")
                                   .astype(np.float32) / 255.0)
            s_task = F.interpolate(imp[None, None], size=(480, 640), mode="bilinear",
                                   align_corners=False).to(device)
            s_hat = torch.sigmoid(F.interpolate(task(ir_mis, vi.mean(1, keepdim=True)),
                                                size=(480, 640), mode="bilinear",
                                                align_corners=False))
            n_e2e = torch.sigmoid(F.interpolate(nec(ir_mis, vi.mean(1, keepdim=True)),
                                                size=(480, 640), mode="bilinear",
                                                align_corners=False))
            out_legacy = legacy(vi, vi, ir_mis)
            neck = out_legacy[7]
            neck = neck[0] if isinstance(neck, (list, tuple)) else neck
            if neck.dim() == 3:
                neck = neck[None]
            neck = neck[:, :1] if neck.shape[1] > 1 else neck
            proxy = F.interpolate(neck, size=(480, 640), mode="bilinear",
                                  align_corners=False).clamp(0, 1)

            g = {
                "full": torch.ones_like(s_task),
                "oracle_geom": build_target(mag[None, None], torch.zeros_like(s_task),
                                            args.geom_scale, 1.0),
                "oracle_task": s_task,
                "oracle_product": build_target(mag[None, None], s_task, args.geom_scale,
                                               args.floor),
                "geom_thresh": (mag[None, None] / args.geom_scale).clamp(0, 1).ge(args.tau).float(),
                "geom_pow": (mag[None, None] / args.geom_scale).clamp(0, 1),
                "task_only": s_hat,
                "task_bin": s_hat.ge(0.5).float(),
                "prod_bin": build_target(mag[None, None], s_hat, args.geom_scale,
                                         args.floor).ge(0.5).float(),
                "factorised": build_target(mag[None, None], s_hat, args.geom_scale, args.floor),
                "e2e": n_e2e,
                "proxy_net": proxy,
            }
            for k, v in g.items():
                acc.setdefault(k, []).append(float(v.mean()))
            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(names)}")

    res = {k: float(np.mean(v)) for k, v in acc.items()}
    args.out.write_text(json.dumps(res, indent=2))
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} mean activation = {v:.4f}")


if __name__ == "__main__":
    main()
