#!/usr/bin/env python3
"""E2: estimate a real registration from the misaligned IR/VIS pair (no GT used).

For each test pair we
  1. reproduce the controlled perturbation (same seed rule as the paper) to obtain
     the misaligned IR and the GT matrix (GT is used ONLY for evaluation metrics),
  2. estimate a global affine warping IR_mis -> VIS with classic multi-modal
     registration: gradient-magnitude images + multi-scale ECC,
  3. store the estimated 2x3 matrix (dst->src convention, i.e. identical to
     `warp_by_matrix(ir_mis, W)` from necessity_common), the ECC correlation and
     the registration error against the GT matrix / GT displacement field.

Output: a .npz with per-image name, W (6), cc, ok, corner error, EPE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
for p in (str(SCRIPTS), str(REPO)):
    sys.path.insert(0, p)

from necessity_common import (  # noqa: E402
    affine_matrices, displacement_field, misalign_ir, sample_affine_params,
)


def load_pair(ir_path: Path, vi_path: Path, h: int, w: int):
    ir = Image.open(ir_path).convert("L").resize((w, h), Image.BILINEAR)
    vi = Image.open(vi_path).convert("RGB").resize((w, h), Image.BILINEAR)
    ir_t = torch.from_numpy(np.asarray(ir, dtype=np.float32) / 255.0)[None, None]
    vi_t = torch.from_numpy(np.asarray(vi, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return ir_t, vi_t


def to_gray_u8(img: torch.Tensor) -> np.ndarray:
    a = img[0].mean(0).clamp(0, 1).numpy()
    return (a * 255).round().astype(np.uint8)


def prep_grad(gray_u8: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gradient magnitude, blurred and contrast-normalised (cross-modal proxy)."""
    g = cv2.GaussianBlur(gray_u8, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), sigma)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    mag = np.clip((mag - lo) / max(hi - lo, 1e-6), 0, 1)
    return (mag * 255).astype(np.uint8)


def phase_corr_init(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Translation-only init: returns a 2x3 matrix sampling `a` at (x - shift)."""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    # b(x) ~ a(x - d)  =>  shift the input by (dx, dy) to align with the template
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def estimate(tpl: np.ndarray, src: np.ndarray, scales=(4, 2, 1)):
    """Multi-scale ECC (euclidean then affine). Returns (W 2x3, cc, ok)."""
    W = np.eye(2, 3, dtype=np.float32)
    # phase-correlation init at coarse scale
    f = scales[0]
    small_t = cv2.resize(tpl, (tpl.shape[1] // f, tpl.shape[0] // f))
    small_s = cv2.resize(src, (src.shape[1] // f, src.shape[0] // f))
    W = phase_corr_init(small_t, small_s) * np.array(
        [[1, 1, f], [1, 1, f]], dtype=np.float32)
    cc, ok = 0.0, False
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    for f in scales:
        h, w = tpl.shape[0] // f, tpl.shape[1] // f
        t = cv2.resize(tpl, (w, h))
        s = cv2.resize(src, (w, h))
        Ws = W.copy()
        Ws[:, 2] /= f
        # euclidean refinement (robust to init error)
        try:
            cc, We = cv2.findTransformECC(t, s, Ws.astype(np.float32),
                                          cv2.MOTION_EUCLIDEAN, criteria, None, 5)
            Ws = We
            ok = True
        except cv2.error:
            pass
        # affine refinement at full resolution only
        if f == 1:
            try:
                cc, Wa = cv2.findTransformECC(t, s, Ws.astype(np.float32),
                                              cv2.MOTION_AFFINE, criteria, None, 5)
                Ws = Wa
                ok = True
            except cv2.error:
                pass
        W = Ws.astype(np.float32)
        W[:, 2] *= f
    return W, float(cc), ok


def displacement_from_W(W: np.ndarray, h: int, w: int) -> torch.Tensor:
    """u(x) = W x - x, magnitude (h,w) float32 (pixel coords, same as GT)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ux = W[0, 0] * xs + W[0, 1] * ys + W[0, 2] - xs
    uy = W[1, 0] * xs + W[1, 1] * ys + W[1, 2] - ys
    return torch.from_numpy(np.sqrt(ux ** 2 + uy ** 2)).float()


def corners_err(W: np.ndarray, GT: np.ndarray, h: int, w: int) -> float:
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]],
                   dtype=np.float64).T
    a = W.astype(np.float64) @ pts
    b = GT.astype(np.float64) @ pts
    return float(np.mean(np.linalg.norm(a - b, axis=0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "sensors-4626757/exp/registration/est_m3fd_test.npz")
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--angle_range", type=float, default=3.0)
    ap.add_argument("--translate_range", type=float, default=8.0)
    ap.add_argument("--scale_range", type=float, default=0.03)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ir_dir = args.data_root / "infrared" / args.split
    vi_dir = args.data_root / "visible" / args.split
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if args.max_samples:
        names = names[:args.max_samples]
    print(f"[est] {len(names)} pairs, {args.img_h}x{args.img_w}")

    rec = []
    t0 = time.time()
    for i, name in enumerate(names):
        ir, vi = load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                           args.img_h, args.img_w)
        params = sample_affine_params(args.angle_range, args.translate_range,
                                      args.scale_range, seed=i)
        ir_mis = misalign_ir(ir, params)
        _, Tinv = affine_matrices(args.img_h, args.img_w, *params)
        tpl = prep_grad(to_gray_u8(vi))
        src = prep_grad(to_gray_u8(ir_mis))
        W, cc, ok = estimate(tpl, src)
        d_est = displacement_from_W(W, args.img_h, args.img_w)
        _, _, mag_gt = displacement_field(args.img_h, args.img_w, *params)
        epe = float((d_est - mag_gt).abs().mean())  # magnitude-field difference
        ce = corners_err(W, Tinv.numpy()[:2, :3], args.img_h, args.img_w)
        rec.append((name, W.ravel(), cc, ok, ce, epe,
                    float(d_est.mean()), float(d_est.max())))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta {el/(i+1)*(len(names)-i-1):.0f}s")

    arr = np.array([[r[1][0], r[1][1], r[1][2], r[1][3], r[1][4], r[1][5]] for r in rec],
                   dtype=np.float32)
    np.savez(args.out,
             names=np.array([r[0] for r in rec]),
             W=arr,
             cc=np.array([r[2] for r in rec], dtype=np.float32),
             ok=np.array([r[3] for r in rec], dtype=bool),
             corner_err=np.array([r[4] for r in rec], dtype=np.float32),
             epe=np.array([r[5] for r in rec], dtype=np.float32),
             d_mean=np.array([r[6] for r in rec], dtype=np.float32),
             d_max=np.array([r[7] for r in rec], dtype=np.float32))
    ce = np.array([r[4] for r in rec])
    ok = np.array([r[3] for r in rec])
    print(f"[est] saved {args.out}")
    print(f"[est] ok rate       : {ok.mean()*100:.1f}%")
    print(f"[est] corner err px : median {np.median(ce):.2f}  mean {ce.mean():.2f}  p90 {np.percentile(ce,90):.2f}  max {ce.max():.2f}")
    print(f"[est] EPE(mag) px    : median {np.median(np.array([r[5] for r in rec])):.2f}")


if __name__ == "__main__":
    main()
