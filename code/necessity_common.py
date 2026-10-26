#!/usr/bin/env python3
"""Shared utilities for the revision (Route B).

Definition of the measured registration-necessity target used everywhere:

    N_gt(x) = normalize( |u_gt(x)|/s_g * (f + (1-f) * S_task(x)) )

where
  * |u_gt(x)| is the ground-truth correction displacement of the synthetic
    perturbation (analytic, see train/dataset.py),
  * s_g is a saturation scale in pixels,
  * S_task(x) in [0,1] is a task-importance map obtained from detection
    annotations (Gaussian-smoothed boxes) and/or a frozen detector,
  * f is the weight floor of the task-importance term.

The registration candidate for the bounded correction is the pixel average
computed with the *registered* infrared image:

    F_pix_mis = 0.5 * (IR_mis + VIS)
    F_pix_reg = 0.5 * (IR_reg + VIS)
    F = F_pix_mis + gamma * N * (F_pix_reg - F_pix_mis)
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import _get_inverse_affine_matrix

IMAGENET_MEAN = 0.5


# ---------------------------------------------------------------- geometry --
def sample_affine_params(angle_range=3.0, translate_range=8.0, scale_range=0.03,
                         seed=None):
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    angle = random.uniform(-angle_range, angle_range)
    translate = [random.uniform(-translate_range, translate_range),
                 random.uniform(-translate_range, translate_range)]
    scale = random.uniform(1.0 - scale_range, 1.0 + scale_range)
    if seed is not None:
        random.setstate(state)
    return angle, translate, scale


def affine_matrices(H, W, angle, translate, scale, dtype=torch.float64):
    """Return (T, T_inv) as 2x3 pixel-coordinate affine matrices.

    T maps output pixel coordinates to the coordinates sampled from the source
    image, i.e. IR_mis(x) = IR(T x) for x given in pixel coordinates.  The
    correction therefore samples IR_mis at T^{-1} x and the correction
    displacement is u(x) = T^{-1} x - x.
    """
    theta = torch.tensor(
        _get_inverse_affine_matrix([0.0, 0.0], angle, list(translate),
                                   scale, [0.0, 0.0]), dtype=dtype).reshape(2, 3)
    lin, off = theta[:, :2], theta[:, 2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    # translate/scale act about the image center
    T = torch.eye(3, dtype=dtype)
    T[:2, :2] = lin
    T[:2, 2] = off
    c = torch.tensor([cx, cy, 1.0], dtype=dtype)
    C = torch.eye(3, dtype=dtype)
    C[:2, 2] = c[:2]
    Cinv = torch.eye(3, dtype=dtype)
    Cinv[:2, 2] = -c[:2]
    T = C @ T @ Cinv
    return T, torch.inverse(T)


def displacement_field(H, W, angle, translate, scale, device="cpu"):
    """Correction displacement (ux, uy) and magnitude, all (H, W) float32."""
    T, Tinv = affine_matrices(H, W, angle, translate, scale)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1)                # (H,W,3)
    Q = P @ Tinv.T                                          # (H,W,3)
    ux = (Q[..., 0] - xs).float()
    uy = (Q[..., 1] - ys).float()
    mag = torch.sqrt(ux ** 2 + uy ** 2)
    return ux.to(device), uy.to(device), mag.to(device)


def warp_by_matrix(img: torch.Tensor, M: torch.Tensor):
    """Sample `img` (B,C,H,W) with pixel-coordinate affine matrix M (2x3)."""
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3)  # (HW,3)
    Q = P @ M.to(img.device, img.dtype).T                    # (HW,3)
    gx = 2.0 * Q[:, 0].reshape(1, H, W) / max(W - 1, 1) - 1.0
    gy = 2.0 * Q[:, 1].reshape(1, H, W) / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).expand(B, -1, -1, -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def misalign_ir(ir, params):
    angle, translate, scale = params
    return TF.affine(ir, angle=angle, translate=list(translate), scale=scale,
                     shear=0, interpolation=TF.InterpolationMode.BILINEAR,
                     fill=0.0)


# ------------------------------------------------------------ task weights --
def boxes_to_importance(boxes, H, W, sigma=12.0, box_scale=1.15):
    """Rasterise YOLO-format boxes to a smooth importance map in [0,1]."""
    imp = np.zeros((H, W), dtype=np.float32)
    if not boxes:
        return imp
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for cx, cy, bw, bh in boxes:
        cxp, cyp = cx * W, cy * H
        half_w = max(bw * W * box_scale * 0.5, 1.0)
        half_h = max(bh * H * box_scale * 0.5, 1.0)
        dx = np.clip(np.abs(xx - cxp) - half_w, 0, None)
        dy = np.clip(np.abs(yy - cyp) - half_h, 0, None)
        imp += np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    return np.clip(imp, 0.0, 1.0)


def load_yolo_boxes(label_path: Path):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        boxes.append((float(cx), float(cy), float(bw), float(bh)))
    return boxes


@torch.no_grad()
def build_target(mag: torch.Tensor, task: torch.Tensor, geom_scale=16.0,
                 floor=0.5) -> torch.Tensor:
    """N_gt = normalize( |u|/geom_scale * (floor + (1-floor) * S_task) )."""
    geom = (mag / float(geom_scale)).clamp(0.0, 1.0)
    t = geom * (float(floor) + (1.0 - float(floor)) * task)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    return ((t - mn) / (mx - mn + 1e-6)).clamp(0.0, 1.0)


# ------------------------------------------------------------------- model --
class ConvGN(nn.Sequential):
    def __init__(self, cin, cout, stride=1, k=3):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(inplace=True),
        )


class NecessityNet(nn.Module):
    """Lightweight misalignment-necessity estimator.

    Input : misaligned IR (1ch) + visible gray (1ch), full resolution.
    Output: necessity map in [0,1] at full resolution.

    The network contains a global-context branch (average pooling) so that the
    overall misalignment level can modulate the local structural response.
    """

    def __init__(self, width=32, mid=64, out_scale=1):
        super().__init__()
        self.out_scale = out_scale
        self.stem = nn.Sequential(
            ConvGN(2, width, stride=2),        # H/2
            ConvGN(width, width, stride=1),
            ConvGN(width, mid, stride=2),      # H/4
            ConvGN(mid, mid, stride=1),
            ConvGN(mid, mid, stride=2),        # H/8
            ConvGN(mid, mid, stride=1),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(mid, mid), nn.SiLU(inplace=True),
            nn.Linear(mid, mid), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            ConvGN(mid, mid, stride=1, k=3),
            nn.Conv2d(mid, 1, 3, 1, 1),
        )

    def forward(self, ir, vi):
        x = torch.cat([ir, vi], dim=1)
        f = self.stem(x)                                   # B,mid,H/8,W/8
        g = f.mean(dim=(2, 3))
        g = self.global_mlp(g).view(-1, f.shape[1], 1, 1)
        f = f * (0.5 + g)                                  # global modulation
        logits = self.head(f)
        return logits


def params_of(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
