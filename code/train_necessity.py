#!/usr/bin/env python3
"""Train the necessity estimator with the measured, verifiable target.

    N_gt(x) = normalize( |u_gt(x)|/s_g * (f + (1-f) * S_task(x)) )

The estimator sees only the misaligned infrared image and the visible image; it
must infer the local registration demand.  Checkpoints are selected by the
Spearman correlation between the predicted map and the ground-truth correction
displacement on a held-out split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from necessity_common import (  # noqa: E402
    NecessityNet, build_target, displacement_field, misalign_ir, params_of,
    sample_affine_params,
)


class NecessityDataset(Dataset):
    def __init__(self, data_root: Path, imp_root: Path, split: str,
                 img_h: int, img_w: int, epoch: int = 0,
                 geom_scale: float = 16.0, floor: float = 0.5,
                 angle_range: float = 3.0, translate_range: float = 8.0,
                 scale_range: float = 0.03, misalign_prob: float = 1.0,
                 fixed_seed: bool = False, noise_sigma: float = 0.0,
                 target_mode: str = "product"):
        self.ir_dir = data_root / "infrared" / split
        self.vi_dir = data_root / "visible" / split
        self.imp_dir = imp_root / split
        self.h, self.w = img_h, img_w
        self.epoch = epoch
        self.geom_scale = geom_scale
        self.floor = floor
        self.ranges = (angle_range, translate_range, scale_range)
        self.misalign_prob = misalign_prob
        self.fixed_seed = fixed_seed
        self.noise_sigma = noise_sigma
        self.target_mode = target_mode
        self.names = sorted(p.stem for p in self.ir_dir.glob("*.png"))
        if not self.names:
            self.names = sorted(p.stem for p in self.ir_dir.glob("*.jpg"))
        assert self.names, f"no images in {self.ir_dir}"

    def __len__(self):
        return len(self.names)

    def _load(self, path: Path, gray: bool):
        im = Image.open(path).convert("L" if gray else "RGB").resize(
            (self.w, self.h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr if gray else arr.mean(axis=2, keepdims=True)

    def __getitem__(self, idx):
        name = self.names[idx]
        ir = torch.from_numpy(self._load(self.ir_dir / f"{name}.png" if
                                         (self.ir_dir / f"{name}.png").exists()
                                         else self.ir_dir / f"{name}.jpg", True))[None]
        vi = torch.from_numpy(self._load(
            self.vi_dir / f"{name}.png" if (self.vi_dir / f"{name}.png").exists()
            else self.vi_dir / f"{name}.jpg", True))[None]

        seed = idx if self.fixed_seed else (self.epoch * 1000003 + idx)
        params = sample_affine_params(*self.ranges, seed=seed)
        if self.misalign_prob >= 1.0 or (idx % 100) < int(self.misalign_prob * 100):
            ir_mis = misalign_ir(ir[None], params)
        else:
            ir_mis = ir[None]
            params = (0.0, [0.0, 0.0], 1.0)
        ir_mis = ir_mis[0]

        if self.noise_sigma > 0:
            g = torch.Generator().manual_seed(seed + 7)
            ir_mis = (ir_mis + torch.randn(ir_mis.shape, generator=g)
                      * self.noise_sigma).clamp(0, 1)
            vi = (vi + torch.randn(vi.shape, generator=g)
                  * self.noise_sigma).clamp(0, 1)

        _, _, mag = displacement_field(self.h, self.w, *params)
        imp_path = self.imp_dir / f"{name}.npy"
        if imp_path.exists():
            imp = torch.from_numpy(
                np.load(imp_path).astype(np.float32) / 255.0)[None, None]
            imp = F.interpolate(imp, size=(self.h, self.w), mode="bilinear",
                                align_corners=False)[0]
        else:
            imp = torch.zeros(1, self.h, self.w)

        if self.target_mode == "task":
            t = imp[None]
            mn = t.amin(dim=(1, 2, 3), keepdim=True)
            mx = t.amax(dim=(1, 2, 3), keepdim=True)
            target = ((t - mn) / (mx - mn + 1e-6)).clamp(0, 1)[0]
        else:
            target = build_target(mag[None, None], imp[None], self.geom_scale,
                                  self.floor)[0]
        return ir_mis, vi, mag[None], target


def pearson_loss(pred, target):
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1)) + 1e-8
    return (1.0 - num / den).mean()


def rank_loss(pred, target, n_pairs=2048, margin=0.05):
    B, _, H, W = pred.shape
    p = pred.flatten(1)
    t = target.flatten(1)
    idx = torch.randint(0, H * W, (B, n_pairs, 2), device=pred.device)
    i, j = idx[..., 0], idx[..., 1]
    ti = torch.gather(t, 1, i)
    tj = torch.gather(t, 1, j)
    sign = torch.sign(ti - tj)
    keep = sign.abs() > 1e-3
    pi = torch.gather(p, 1, i)
    pj = torch.gather(p, 1, j)
    diff = pi - pj
    loss = torch.clamp(margin - sign * diff, min=0.0)
    return (loss * keep).sum() / (keep.sum() + 1e-6)


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


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


@torch.no_grad()
def validate(model, loader, device, stride=4, max_images=200):
    model.eval()
    rhos, aucs = [], []
    for k, (ir_mis, vi, mag, target) in enumerate(loader):
        ir_mis, vi = ir_mis.to(device), vi.to(device)
        logits = model(ir_mis, vi)
        pred = torch.sigmoid(F.interpolate(
            logits, size=mag.shape[-2:], mode="bilinear", align_corners=False))
        for b in range(pred.shape[0]):
            p = pred[b, 0, ::stride, ::stride].cpu().numpy().ravel()
            # select against the actual supervision target of this run
            m = target[b, 0, ::stride, ::stride].numpy().ravel()
            rhos.append(spearman(p, m))
            thr = np.quantile(m, 0.75)
            lab = (m >= thr).astype(np.int64)
            pos, neg = p[lab == 1], p[lab == 0]
            if len(pos) and len(neg):
                allv = np.concatenate([pos, neg])
                order = np.argsort(allv)
                ranks = np.empty(len(allv))
                ranks[order] = np.arange(1, len(allv) + 1)
                rpos = ranks[:len(pos)].sum()
                aucs.append((rpos - len(pos) * (len(pos) + 1) / 2) /
                            (len(pos) * len(neg)))
        if k + 1 >= max_images // loader.batch_size:
            break
    return float(np.nanmean(rhos)), float(np.nanmean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/ubuntu/Code/datasets/M3FD_rebuilt"))
    ap.add_argument("--imp_root", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                                 "revision_2026-09/cache/task_importance/M3FD"))
    ap.add_argument("--output", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                                 "revision_2026-09/results/necessity_v1"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--lambda_corr", type=float, default=0.5)
    ap.add_argument("--lambda_rank", type=float, default=0.2)
    ap.add_argument("--misalign_prob", type=float, default=1.0)
    ap.add_argument("--angle_range", type=float, default=3.0)
    ap.add_argument("--translate_range", type=float, default=8.0)
    ap.add_argument("--scale_range", type=float, default=0.03)
    ap.add_argument("--target", type=str, default="product",
                    choices=["product", "task"],
                    help="product: measured necessity target; "
                         "task: task-importance map only")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--mid", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    model = NecessityNet(width=args.width, mid=args.mid).to(device)
    print(f"[necessity] trainable params: {params_of(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    val_ds = NecessityDataset(args.data_root, args.imp_root, "test",
                              args.img_h, args.img_w, fixed_seed=True,
                              geom_scale=args.geom_scale, floor=args.floor,
                              angle_range=args.angle_range,
                              translate_range=args.translate_range,
                              scale_range=args.scale_range,
                              misalign_prob=1.0, target_mode=args.target)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    log_path = args.output / "train_log.csv"
    best_rho = -1.0
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "l1", "corr", "rank",
                         "val_spearman", "val_auc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            ds = NecessityDataset(args.data_root, args.imp_root, "train",
                                  args.img_h, args.img_w, epoch=epoch,
                                  geom_scale=args.geom_scale, floor=args.floor,
                                  angle_range=args.angle_range,
                                  translate_range=args.translate_range,
                                  scale_range=args.scale_range,
                                  misalign_prob=args.misalign_prob,
                                  target_mode=args.target)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True,
                                drop_last=True)
            model.train()
            t0 = time.time()
            agg = np.zeros(4)
            n_rollback = 0
            for ir_mis, vi, mag, target in loader:
                ir_mis, vi = ir_mis.to(device), vi.to(device)
                target = target.to(device)
                logits = model(ir_mis, vi)
                pred = torch.sigmoid(F.interpolate(
                    logits, size=target.shape[-2:], mode="bilinear",
                    align_corners=False))
                l1 = F.l1_loss(pred, target)
                lc = pearson_loss(pred, target)
                lr_ = rank_loss(pred, target)
                loss = l1 + args.lambda_corr * lc + args.lambda_rank * lr_
                if not torch.isfinite(loss):
                    # keep training stable on degenerate batches
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grads_ok = all(torch.isfinite(p.grad).all()
                               for p in model.parameters() if p.grad is not None)
                if not grads_ok:
                    opt.zero_grad(set_to_none=True)
                    continue
                snapshot = [p.detach().clone() for p in model.parameters()]
                opt.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    with torch.no_grad():
                        for p, s in zip(model.parameters(), snapshot):
                            p.copy_(s)
                    n_rollback += 1
                agg += [float(loss), float(l1), float(lc), float(lr_)]
            sched.step()
            agg /= max(len(loader), 1)
            rho, auc = validate(model, val_loader, device)
            dt = time.time() - t0
            writer.writerow([epoch, f"{agg[0]:.5f}", f"{agg[1]:.5f}",
                             f"{agg[2]:.5f}", f"{agg[3]:.5f}",
                             f"{rho:.4f}", f"{auc:.4f}", f"{dt:.0f}"])
            f.flush()
            print(f"[Ep {epoch:3d}/{args.epochs}] loss={agg[0]:.4f} "
                  f"l1={agg[1]:.4f} corr={agg[2]:.4f} rank={agg[3]:.4f} | "
                  f"val_rho={rho:.3f} val_auc={auc:.3f} | {dt:.0f}s"
                  f"{' rollback=' + str(n_rollback) if n_rollback else ''}")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_spearman": rho, "val_auc": auc,
                        "args": vars(args)},
                       args.output / "last.pt")
            if rho > best_rho:
                best_rho = rho
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_spearman": rho, "val_auc": auc,
                            "args": vars(args)},
                           args.output / "best.pt")
    (args.output / "summary.json").write_text(json.dumps({
        "best_val_spearman": best_rho,
        "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))


if __name__ == "__main__":
    main()
