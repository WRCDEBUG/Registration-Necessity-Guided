#!/usr/bin/env python3
"""E2 v2: bounded multi-modal affine registration, matched to the perturbation class.

Rationale for v2 (why v1 failed): unconstrained ECC drifted in scale and, for some
pairs, in translation (corner error p90 = 91 px). A deployed system that knows the
*class* of misalignment (small rotation, small translation, small scale) would use a
bounded, coarse-to-fine search instead. We therefore estimate

    theta = (rotation, tx, ty, scale)          with hard bounds
    W = Tinv(theta)   (dst->src, same convention as warp_by_matrix)

by maximising the normalised cross-correlation of gradient-magnitude images
(cross-modal proxy) over a coarse grid + Nelder-Mead refinement.

GT is never used inside the estimation: the bounds come from the acquisition prior
documented in the paper (|rot| <= 4 deg, |trans| <= 10 px, scale in [0.96, 1.04]),
which is a *superset* of the reported perturbation range.

Output npz: names, W (6), cc, ok, corner_err, epe, d_mean, d_max
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(1)  # affine_matrices builds tiny tensors; threading is pure overhead
from PIL import Image
from scipy.optimize import minimize

REPO = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register")
SCRIPTS = REPO / "过程文件" / "revision_2026-09" / "scripts"
for p in (str(SCRIPTS), str(REPO)):
    sys.path.insert(0, p)

from necessity_common import (  # noqa: E402
    affine_matrices, displacement_field, misalign_ir, sample_affine_params,
)

BOUNDS = [(4.0, -4.0), (-10.0, 10.0), (-10.0, 10.0), (1.04, 0.96)]  # rot, tx, ty, scale


def load_pair(ir_path: Path, vi_path: Path, h: int, w: int):
    ir = Image.open(ir_path).convert("L").resize((w, h), Image.BILINEAR)
    vi = Image.open(vi_path).convert("RGB").resize((w, h), Image.BILINEAR)
    ir_t = torch.from_numpy(np.asarray(ir, dtype=np.float32) / 255.0)[None, None]
    vi_t = torch.from_numpy(np.asarray(vi, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return ir_t, vi_t


def to_gray(img: torch.Tensor) -> np.ndarray:
    a = img[0].mean(0).clamp(0, 1).numpy()
    return (a * 255).round().astype(np.uint8)


def grad(gray_u8: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    g = cv2.GaussianBlur(gray_u8, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), sigma)
    m -= m.mean()
    s = m.std()
    return (m / (s + 1e-6)).astype(np.float32)


def scale_matrix(M: np.ndarray, f: float) -> np.ndarray:
    """Rescale a full-res dst->src matrix for a 1/f downsampled image pair."""
    S = np.diag([1.0 / f, 1.0 / f, 1.0])
    Si = np.diag([f, f, 1.0])
    M3 = np.eye(3, dtype=np.float64)
    M3[:2] = M
    return (S @ M3 @ Si)[:2]


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    va, vb = a[mask], b[mask]
    va = va - va.mean()
    vb = vb - vb.mean()
    den = np.sqrt((va * va).sum() * (vb * vb).sum()) + 1e-9
    return float((va * vb).sum() / den)


def make_objective(tpl_g, src_g, h, w, f, mask):
    def obj(theta):
        rot, tx, ty, sc = theta
        _, Tinv = affine_matrices(h, w, rot, (tx, ty), sc)
        M = scale_matrix(Tinv.numpy()[:2, :3], f)
        warped = cv2.warpAffine(src_g, M, (w // f, h // f),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        return -ncc(warped, tpl_g, mask)
    return obj


def pattern_search(obj, theta, steps, rounds=6):
    """Deterministic coordinate search with decreasing steps (bounded)."""
    theta = np.array(theta, dtype=np.float64)
    best = obj(theta)
    for step in steps:
        improved = True
        r = 0
        while improved and r < rounds:
            improved = False
            r += 1
            for k in range(4):
                for sgn in (+1.0, -1.0):
                    cand = theta.copy()
                    cand[k] += sgn * step[k]
                    cand = np.clip(cand, [b[1] for b in BOUNDS], [b[0] for b in BOUNDS])
                    v = obj(cand)
                    if v < best - 1e-7:
                        best, theta, improved = v, cand, True
    return theta, best


def estimate(tpl_u8: np.ndarray, src_u8: np.ndarray, h: int, w: int):
    f = 4
    tpl_g = grad(cv2.resize(tpl_u8, (w // f, h // f)))
    src_g = grad(cv2.resize(src_u8, (w // f, h // f)))
    mask = np.zeros_like(tpl_g, dtype=bool)
    b = f + 1
    mask[b:h // f - b, b:w // f - b] = True
    obj = make_objective(tpl_g, src_g, h, w, f, mask)

    best, best_v = (0.0, 0.0, 0.0, 1.0), 1e9
    for rot in (-3.0, -1.5, 0.0, 1.5, 3.0):
        for sc in (0.97, 1.0, 1.03):
            for tx in (-8.0, -4.0, 0.0, 4.0, 8.0):
                for ty in (-8.0, -4.0, 0.0, 4.0, 8.0):
                    v = obj((rot, tx, ty, sc))
                    if v < best_v:
                        best_v, best = v, (rot, tx, ty, sc)

    # refine at half resolution with a pattern search
    f = 2
    tpl_g2 = grad(cv2.resize(tpl_u8, (w // f, h // f)))
    src_g2 = grad(cv2.resize(src_u8, (w // f, h // f)))
    mask2 = np.zeros_like(tpl_g2, dtype=bool)
    mask2[3:h // f - 3, 3:w // f - 3] = True
    obj2 = make_objective(tpl_g2, src_g2, h, w, f, mask2)
    theta2, v2 = pattern_search(obj2, best,
                                [(0.6, 1.5, 1.5, 0.008),
                                 (0.25, 0.5, 0.5, 0.003),
                                 (0.1, 0.2, 0.2, 0.001)])
    # final polish at full resolution
    tpl_g1 = grad(tpl_u8)
    src_g1 = grad(src_u8)
    mask1 = np.zeros_like(tpl_g1, dtype=bool)
    mask1[2:-2, 2:-2] = True
    obj1 = make_objective(tpl_g1, src_g1, h, w, 1, mask1)
    theta1, v1 = pattern_search(obj1, theta2,
                                [(0.1, 0.3, 0.3, 0.001), (0.04, 0.1, 0.1, 0.0004)],
                                rounds=3)
    theta, cc = (theta1, -v1) if v1 < v2 else (theta2, -v2)
    _, Tinv = affine_matrices(h, w, float(theta[0]),
                              (float(theta[1]), float(theta[2])), float(theta[3]))
    return Tinv.numpy()[:2, :3].astype(np.float32), float(cc), True, theta


def displacement_from_W(W: np.ndarray, h: int, w: int) -> torch.Tensor:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ux = W[0, 0] * xs + W[0, 1] * ys + W[0, 2] - xs
    uy = W[1, 0] * xs + W[1, 1] * ys + W[1, 2] - ys
    return torch.from_numpy(np.sqrt(ux ** 2 + uy ** 2)).float()


def corners_err(W: np.ndarray, GT: np.ndarray, h: int, w: int) -> float:
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]]).T
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
                    default=REPO / "sensors-4626757/exp/registration/est_v2.npz")
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
    print(f"[est2] {len(names)} pairs")

    rec, t0 = [], time.time()
    for i, name in enumerate(names):
        ir, vi = load_pair(ir_dir / f"{name}.png", vi_dir / f"{name}.png",
                           args.img_h, args.img_w)
        params = sample_affine_params(args.angle_range, args.translate_range,
                                      args.scale_range, seed=i)
        ir_mis = misalign_ir(ir, params)
        _, Tinv = affine_matrices(args.img_h, args.img_w, *params)
        W, cc, ok, theta = estimate(to_gray(vi), to_gray(ir_mis),
                                    args.img_h, args.img_w)
        d_est = displacement_from_W(W, args.img_h, args.img_w)
        _, _, mag_gt = displacement_field(args.img_h, args.img_w, *params)
        epe = float((d_est - mag_gt).abs().mean())
        ce = corners_err(W, Tinv.numpy()[:2, :3], args.img_h, args.img_w)
        rec.append((name, W.ravel(), cc, ok, ce, epe,
                    float(d_est.mean()), float(d_est.max()), *theta))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta {el/(i+1)*(len(names)-i-1):.0f}s")

    arr = np.array([r[1] for r in rec], dtype=np.float32)
    np.savez(args.out,
             names=np.array([r[0] for r in rec]), W=arr,
             cc=np.array([r[2] for r in rec], np.float32),
             ok=np.array([r[3] for r in rec], bool),
             corner_err=np.array([r[4] for r in rec], np.float32),
             epe=np.array([r[5] for r in rec], np.float32),
             d_mean=np.array([r[6] for r in rec], np.float32),
             d_max=np.array([r[7] for r in rec], np.float32),
             theta=np.array([[r[8], r[9], r[10], r[11]] for r in rec], np.float32))
    ce = np.array([r[4] for r in rec])
    epe = np.array([r[5] for r in rec])
    print(f"[est2] saved {args.out}")
    print(f"[est2] corner err px: median {np.median(ce):.2f} mean {ce.mean():.2f} "
          f"p90 {np.percentile(ce, 90):.2f} max {ce.max():.2f}")
    print(f"[est2] EPE(mag) px  : median {np.median(epe):.2f} mean {epe.mean():.2f}")
    print(f"[est2] within 2 px  : {(ce <= 2).mean()*100:.1f}%   within 4 px: {(ce <= 4).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
