#!/usr/bin/env python3
"""Third real registration module: RAFT (learned optical flow) + robust affine (E2c).

Reviewer round 2 asked for at least one *learned* cross-modal registrar. This module
runs pretrained RAFT (torchvision, C-T-SKHT-V2 weights) on the misaligned IR / VIS
pair, converts the dense flow into correspondences on a coarse grid, and fits a
partial affine with RANSAC. Output matches est_sift.py so eval_est_reg.py can be
reused unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
sys.path.insert(0, str(REPO / "过程文件/revision_2026-09/scripts"))

from necessity_common import affine_matrices, misalign_ir, sample_affine_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/est_raft_test.npz")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--min_inliers", type=int, default=200)
    args = ap.parse_args()

    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().cuda()
    torch.set_num_threads(1)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if args.max_samples:
        names = names[:args.max_samples]
    H, W = args.img_h, args.img_w

    Ws, ccs, oks, cerr, epes = [], [], [], [], []
    with torch.no_grad():
        for idx, name in enumerate(names):
            ir = cv2.imread(str(ir_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            vi = cv2.imread(str(vi_dir / f"{name}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
            ir = cv2.resize(ir, (W, H), interpolation=cv2.INTER_AREA)
            vi = cv2.resize(vi, (W, H), interpolation=cv2.INTER_AREA)
            ir_t = torch.from_numpy(ir.astype(np.float32) / 255.0)[None, None]
            params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
            mis = misalign_ir(ir_t, params)[0, 0].numpy()
            _, Tinv = affine_matrices(H, W, *params)
            a = torch.from_numpy(np.repeat(mis[None, None], 3, 1).astype(np.float32) / 255.0).cuda()
            b = torch.from_numpy(vi.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).cuda()
            flow = model(a, b)[-1][0].cpu().numpy()

            s = args.stride
            ys, xs = np.mgrid[s // 2:H - s // 2:s, s // 2:W - s // 2:s]
            src = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
            fx = flow[0][ys, xs].ravel()
            fy = flow[1][ys, xs].ravel()
            dst = src + np.stack([fx, fy], 1)
            M, mask = cv2.estimateAffinePartial2D(
                src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.RANSAC,
                ransacReprojThreshold=6.0, maxIters=3000, confidence=0.99)
            inl = int(mask.sum()) if mask is not None else 0
            ok = M is not None and inl >= args.min_inliers
            if ok:
                M3 = np.eye(3)
                M3[:2] = M
                sc = float(np.sqrt(abs(np.linalg.det(M3))))
                ok = 0.85 < sc < 1.18
            if ok:
                Winv = np.linalg.inv(M3)[:2]
            else:
                Winv = np.eye(2, 3)
            Ws.append(Winv.reshape(-1))

            G = Tinv.numpy()[:2].astype(np.float64)
            corn = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]]).T
            ce = np.linalg.norm((Winv.astype(np.float64) @ corn) - (G @ corn), axis=0).mean()
            yy, xx = np.mgrid[0:H:32, 0:W:32]
            pts = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
            epe = np.linalg.norm((Winv.astype(np.float64) @ pts) - (G @ pts), axis=0).mean()
            ccs.append(inl / max(len(src), 1))
            oks.append(ok)
            cerr.append(ce)
            epes.append(epe)
            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(names)} ok={np.mean(oks):.2f} "
                      f"corner_med={np.median(np.array(cerr)[np.array(oks, bool)]):.2f}")

    np.savez(args.out, names=np.array(names), W=np.array(Ws, np.float32),
             cc=np.array(ccs, np.float32), ok=np.array(oks, np.float32),
             corner_err=np.array(cerr, np.float32), epe=np.array(epes, np.float32))
    ok = np.array(oks, bool)
    print(f"[raft] {len(names)} pairs, success {ok.mean():.3f}")
    if ok.any():
        print(f"[raft] corner err (ok only): median {np.median(np.array(cerr)[ok]):.2f} px")
        print(f"[raft] EPE (ok only): median {np.median(np.array(epes)[ok]):.2f} px")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
