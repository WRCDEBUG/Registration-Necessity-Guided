#!/usr/bin/env python3
"""Evaluation of the bounded necessity-gated correction pipeline.

For every test pair the script builds

    F_pix_mis = 0.5 (IR_mis + VIS)          (no correction)
    F_pix_reg = 0.5 (IR_reg + VIS)          (full correction, GT-registered IR)
    F_gate    = F_pix_mis + gamma * N * (F_pix_reg - F_pix_mis)

for a set of gate maps N (oracle geometric, oracle task, oracle product,
learned, uniform, shuffled, random) and gamma values, saves the resulting
images, computes necessity-validity statistics, and optionally runs the YOLO
detector on every condition.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision_2026-09" / "scripts"))

from eval.eval_tdsr import run_yolo_eval  # noqa: E402
from necessity_common import (  # noqa: E402
    NecessityNet, affine_matrices, build_target, displacement_field,
    misalign_ir, sample_affine_params, warp_by_matrix,
)
from exp_a_necessity_validity import auc_score, spearman  # noqa: E402


def rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
        i = j + 1
    return ranks


def _load_pair(ir_path: Path, vi_path: Path, h: int, w: int, device):
    ir = Image.open(ir_path).convert("L").resize((w, h), Image.BILINEAR)
    vi = Image.open(vi_path).convert("RGB").resize((w, h), Image.BILINEAR)
    ir_t = torch.from_numpy(np.asarray(ir, dtype=np.float32) / 255.0)[None, None]
    vi_t = torch.from_numpy(np.asarray(vi, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return ir_t.to(device), vi_t.to(device)


def save_img(t: torch.Tensor, path: Path) -> None:
    arr = t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imp_root", type=Path,
                    default=ROOT / "revision_2026-09/cache/task_importance/M3FD")
    ap.add_argument("--label_dir", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test"))
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="trained NecessityNet checkpoint (optional)")
    ap.add_argument("--task_ckpt", type=Path, default=None,
                    help="trained task-importance checkpoint; the factorised "
                         "gate is d_hat * (floor + (1-floor) * S_hat)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "revision_2026-09/results/gated_eval")
    ap.add_argument("--yolo", type=str, default=str(ROOT / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--gates", type=str,
                    default="oracle_product,oracle_geom,oracle_task,uniform,shuffled,random")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--angle_range", type=float, default=3.0)
    ap.add_argument("--translate_range", type=float, default=8.0)
    ap.add_argument("--scale_range", type=float, default=0.03)
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--geom_noise", type=float, default=0.0,
                    help="relative error applied to the geometric factor "
                         "(emulates registration-estimate error)")
    ap.add_argument("--skip_yolo", action="store_true")
    ap.add_argument("--gen_only", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    conditions = ["none", "full"]
    for g in gates:
        for gam in args.gammas:
            conditions.append(f"{g}_g{gam:g}")

    for c in conditions:
        (args.output / "images" / c).mkdir(parents=True, exist_ok=True)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if not names:
        names = sorted(p.stem for p in ir_dir.glob("*.jpg"))
    if args.max_samples:
        names = names[:args.max_samples]

    model = None
    if args.ckpt is not None and Path(args.ckpt).exists():
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        model = NecessityNet().to(device).eval()
        model.load_state_dict(sd)
        print(f"[eval] loaded necessity checkpoint {args.ckpt}")

    task_model = None
    if args.task_ckpt is not None and Path(args.task_ckpt).exists():
        ck = torch.load(args.task_ckpt, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        task_model = NecessityNet().to(device).eval()
        task_model.load_state_dict(sd)
        print(f"[eval] loaded task-importance checkpoint {args.task_ckpt}")

    rows = []
    stats = {"rho_geom": [], "rho_task": [], "rho_product": [], "auc_geom": []}
    print(f"[eval] {len(names)} samples, {len(conditions)} conditions")

    for idx, name in enumerate(names):
        ir, vi = _load_pair(ir_dir / f"{name}.png"
                            if (ir_dir / f"{name}.png").exists()
                            else ir_dir / f"{name}.jpg",
                            vi_dir / f"{name}.png"
                            if (vi_dir / f"{name}.png").exists()
                            else vi_dir / f"{name}.jpg",
                            args.img_h, args.img_w, device)
        vi_gray = vi.mean(dim=1, keepdim=True)

        params = sample_affine_params(args.angle_range, args.translate_range,
                                      args.scale_range, seed=idx)
        ir_mis = misalign_ir(ir, params)
        _, _, mag = displacement_field(args.img_h, args.img_w, *params, device=device)
        _, Tinv = affine_matrices(args.img_h, args.img_w, *params)
        ir_reg = warp_by_matrix(ir_mis, Tinv.to(device).float())

        ir_rgb = ir_mis.repeat(1, 3, 1, 1)
        reg_rgb = ir_reg.repeat(1, 3, 1, 1)
        f_mis = (0.5 * ir_rgb + 0.5 * vi).clamp(0, 1)
        f_reg = (0.5 * reg_rgb + 0.5 * vi).clamp(0, 1)
        delta = f_reg - f_mis

        imp_path = args.imp_root / args.split / f"{name}.npy"
        if imp_path.exists():
            imp = torch.from_numpy(np.load(imp_path).astype(np.float32) / 255.0)
            task = F.interpolate(imp[None, None], size=(args.img_h, args.img_w),
                                 mode="bilinear", align_corners=False)[0]
        else:
            task = torch.zeros(1, args.img_h, args.img_w, device=device)
        task = task.to(device)

        mag01 = (mag[None, None] / args.geom_scale).clamp(0, 1)
        oracle_product = build_target(mag[None, None], task[None], args.geom_scale,
                                      args.floor)[0]
        oracle_geom = build_target(mag[None, None],
                                   torch.zeros_like(task)[None], args.geom_scale,
                                   1.0)[0]
        oracle_task = task

        # validity statistics (subsampled)
        sel = torch.zeros(args.img_h, args.img_w, dtype=torch.bool, device=device)
        sel[::3, ::3] = True
        def flat(t):
            t = t.squeeze()
            if t.dim() == 3:
                t = t[0]
            return t[sel].detach().cpu().numpy()

        gates_map = {
            "oracle_product": oracle_product,
            "oracle_geom": oracle_geom,
            "oracle_task": oracle_task,
            "uniform": torch.full_like(oracle_product, float(oracle_product.mean())),
            "shuffled": oracle_product.flatten()[
                torch.randperm(oracle_product.numel(), device=device)].view_as(oracle_product),
            "random": torch.from_numpy(
                rng.random((1, args.img_h, args.img_w)).astype(np.float32)).to(device),
        }
        src_ir, src_vi = ir_mis, vi_gray
        if args.noise_sigma > 0:
            g = torch.Generator(device=device).manual_seed(idx + 11)
            src_ir = (src_ir + torch.randn(src_ir.shape, generator=g,
                                           device=device) * args.noise_sigma).clamp(0, 1)
            src_vi = (src_vi + torch.randn(src_vi.shape, generator=g,
                                           device=device) * args.noise_sigma).clamp(0, 1)

        if model is not None:
            logits = model(src_ir, src_vi)
            learned = torch.sigmoid(F.interpolate(
                logits, size=(args.img_h, args.img_w), mode="bilinear",
                align_corners=False))[0]
            gates_map["learned"] = learned

        if task_model is not None:
            logits = task_model(src_ir, src_vi)
            s_hat = torch.sigmoid(F.interpolate(
                logits, size=(args.img_h, args.img_w), mode="bilinear",
                align_corners=False))
            mag_for_gate = mag[None, None]
            if args.geom_noise > 0:
                g = torch.Generator(device=device).manual_seed(idx + 101)
                eps = 1.0 + args.geom_noise * torch.randn(
                    (1, 1, 8, 8), generator=g, device=device)
                eps = F.interpolate(eps, size=(args.img_h, args.img_w),
                                    mode="bilinear", align_corners=False)
                mag_for_gate = mag_for_gate * eps
            gates_map["learned_factorized"] = build_target(
                mag_for_gate, s_hat, args.geom_scale, args.floor)[0]

        rows.append(dict(
            name=name,
            rho_geom=spearman(flat(oracle_geom), flat(mag01)),
            rho_product=spearman(flat(oracle_product), flat(mag01)),
            auc_geom=auc_score(flat(oracle_geom), (flat(mag01) >= 0.5).astype(np.int64)),
            mean_geom=float(mag01.mean()), max_geom=float(mag01.max()),
        ))

        save_img(f_mis, args.output / "images" / "none" / f"{name}.png")
        save_img(f_reg, args.output / "images" / "full" / f"{name}.png")
        for g, gate in gates_map.items():
            if g not in gates:
                continue
            for gam in args.gammas:
                out = (f_mis + gam * gate[None, ...] * delta).clamp(0, 1)
                save_img(out, args.output / "images" / f"{g}_g{gam:g}" / f"{name}.png")

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(names)}")

    summary = {"n_samples": len(names),
               "gates": gates, "gammas": args.gammas,
               "mean_geom_displacement_px": float(np.mean([r["mean_geom"] for r in rows]) * args.geom_scale)}
    (args.output / "validity_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.skip_yolo and not args.gen_only:
        results = {}
        for c in conditions:
            d = args.output / "images" / c
            if not any(d.glob("*.png")):
                continue
            print(f"  YOLO: {c}")
            results[c] = run_yolo_eval(d, args.yolo, args.label_dir,
                                       args.img_h, args.img_w)
        (args.output / "detection_results.json").write_text(
            json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
