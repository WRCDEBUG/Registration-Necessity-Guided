#!/usr/bin/env python3
"""Second real registration module: SIFT + RANSAC (E2b).

The reviewer named RAFT/STN/MURF as examples of real registration modules. The
paper's E2a already replaces the oracle geometry with a classical estimator
(gradient correlation + bounded search). This script adds an *independent*
implementation, a feature-based SIFT + RANSAC matcher, so the conclusion can be
checked against a second, structurally different estimator.

Output: npz with `names`, `W` (2x3 inverse-warp matrices, same convention as
eval_est_reg.py), `cc` (inlier ratio), `ok`, `corner_err`, `epe`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from necessity_common import affine_matrices, misalign_ir, sample_affine_params  # noqa: E402


def to_u8(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return (g * 255).astype(np.uint8)


def grad_u8(g: np.ndarray) -> np.ndarray:
    """SIFT works better on gradient magnitude for this cross-modal pair."""
    u = to_u8(g).astype(np.float32)
    gm = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) ** 2 + \
         cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3) ** 2
    return to_u8(np.sqrt(gm))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/est_sift_test.npz")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--min_inliers", type=int, default=6)
    args = ap.parse_args()

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if args.max_samples:
        names = names[:args.max_samples]
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    H, W = args.img_h, args.img_w

    Ws, ccs, oks, cerr, epes = [], [], [], [], []
    for idx, name in enumerate(names):
        ir = cv2.imread(str(ir_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        vi = cv2.imread(str(vi_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        ir = cv2.resize(ir, (W, H), interpolation=cv2.INTER_AREA)
        vi = cv2.resize(vi, (W, H), interpolation=cv2.INTER_AREA)

        ir_t = torch.from_numpy(ir.astype(np.float32) / 255.0)[None, None]
        params = sample_affine_params(3.0, 8.0, 0.03, seed=idx)
        ir_mis = misalign_ir(ir_t, params)[0, 0].numpy()
        _, Tinv = affine_matrices(H, W, *params)

        kp1, d1 = sift.detectAndCompute(grad_u8(ir_mis), None)
        kp2, d2 = sift.detectAndCompute(grad_u8(vi), None)
        M = None
        inl = 0
        if d1 is not None and d2 is not None and len(kp1) > 8 and len(kp2) > 8:
            matches = matcher.knnMatch(d1, d2, k=2)
            good = [m for m, n in (mm for mm in matches if len(mm) == 2)
                    if m.distance < 0.85 * n.distance]
            if len(good) >= args.min_inliers:
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                    maxIters=5000, confidence=0.995)
                if M is not None and mask is not None:
                    inl = int(mask.sum())
                    M3c = np.eye(3)
                    M3c[:2] = M
                    sc = float(np.sqrt(abs(np.linalg.det(M3c))))
                    if inl < args.min_inliers or not (0.85 < sc < 1.18):
                        M = None
                else:
                    M = None
        ok = M is not None
        # inverse-warp convention: ir_reg(x) = ir_mis(M^-1 x)
        if ok:
            M3 = np.eye(3, dtype=np.float64)
            M3[:2] = M
            Winv = np.linalg.inv(M3)[:2]
        else:
            Winv = np.eye(2, 3)
        Ws.append(Winv.reshape(-1))

        # error metrics against the ground-truth correction
        G = Tinv.numpy()[:2].astype(np.float64)
        corn = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]]).T
        ce = np.linalg.norm((Winv @ corn) - (G @ corn), axis=0).mean()
        ys, xs = np.mgrid[0:H:32, 0:W:32]
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
        epe = np.linalg.norm((Winv @ pts) - (G @ pts), axis=0).mean()
        ccs.append(inl / max(len(kp1), 1))
        oks.append(ok)
        cerr.append(ce)
        epes.append(epe)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(names)}  ok={np.mean(oks):.2f} "
                  f"corner_err_med={np.median(cerr):.2f} epe_med={np.median(epes):.2f}")

    np.savez(args.out, names=np.array(names), W=np.array(Ws, np.float32),
             cc=np.array(ccs, np.float32), ok=np.array(oks, np.float32),
             corner_err=np.array(cerr, np.float32), epe=np.array(epes, np.float32))
    ok = np.array(oks, bool)
    print(f"[sift] {len(names)} pairs, success {ok.mean():.3f}")
    print(f"[sift] corner err (ok only): median {np.median(np.array(cerr)[ok]):.2f} px")
    print(f"[sift] EPE (ok only): median {np.median(np.array(epes)[ok]):.2f} px")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
