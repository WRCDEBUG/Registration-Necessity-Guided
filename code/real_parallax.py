#!/usr/bin/env python3
"""MS2 real-parallax experiment: is gating a *global* registration worthwhile?

MS2 gives a visible (RGB) and a thermal camera with a large baseline plus LiDAR
depth for both, so the RGB-thermal misalignment is a genuine depth-dependent
parallax field. Geometry was validated before use: RGB depth points project into
the thermal frame for 100 % of points, the two projected depth maps agree to
0.06 m median, and the round-trip correspondence error is 0.31 px median.

The deployment question is not "is a dense field better than nothing" (it is), but
"given a *global* registration model, should it be applied everywhere or only where
the parallax residual is small?"  Conditions, composited in the thermal frame as
the pixel average with the thermal image:

  raw           : RGB resized to the thermal frame (no registration)
  H_dense       : global homography (depth-independent rotation) applied everywhere
  H_gate_far    : H applied only where the parallax residual is small, d < tau
                  (the sensible gate: trust the global model in the far field)
  H_gate_near   : H applied only where d > tau (control: the wrong allocation)
  H_soft_far    : soft variant, N = clip(1 - d / s_g, 0, 1)
  full          : depth-based per-pixel warp (oracle upper reference)

d is the true parallax residual between the global model and the depth field, so
H_gate/H_soft are oracle-geometry gates (the paper's oracle-geometry condition).

Metrics need no labels (the released package has none): mutual information between
the warped RGB and the thermal image, reported globally and stratified by the true
parallax (near field d > 20 px, far field d < 5 px), plus sharpness and entropy of
the fused output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/home/ubuntu/Code/datasets/MS2_300帧/extracted")
OUT_DEFAULT = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                   "sensors-4626757/exp/ms2/results")


def load_calib(seq_dir: Path):
    c = np.load(next(seq_dir.glob("sync_data/*/calib.npy")), allow_pickle=True).item()

    def ext(R, T):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = np.asarray(T).reshape(3) / 1000.0
        return M

    E_rgb = ext(c["R_nir2rgb"], c["T_nir2rgb"])
    E_thr = ext(c["R_nir2thr"], c["T_nir2thr"])
    M_r2t = E_thr @ np.linalg.inv(E_rgb)
    return c["K_rgbL"], c["K_thrL"], np.linalg.inv(M_r2t)


def dense_depth(dt: np.ndarray) -> np.ndarray:
    valid = (dt > 0).astype(np.uint8)
    if valid.sum() == 0:
        return np.full_like(dt, 50.0)
    _, labels = cv2.distanceTransformWithLabels(1 - valid, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    idx = np.flatnonzero(valid.ravel())
    look = np.zeros(labels.max() + 1, np.float32)
    look[labels.ravel()[idx]] = dt.ravel()[idx]
    dense = cv2.medianBlur(look[labels], 5)
    dense = np.where(valid > 0, dt, dense)
    return np.maximum(dense, 0.5)


def mappings(K_rgb, K_thr, M_t2r, Z):
    H, W = Z.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dirs = np.stack([(xs - K_thr[0, 2]) / K_thr[0, 0],
                     (ys - K_thr[1, 2]) / K_thr[1, 1],
                     np.ones_like(xs)], -1)
    R, t = M_t2r[:3, :3], M_t2r[:3, 3]
    dr = dirs @ R.T
    u_inf = (K_rgb[0, 0] * dr[..., 0] / dr[..., 2] + K_rgb[0, 2]).astype(np.float32)
    v_inf = (K_rgb[1, 1] * dr[..., 1] / dr[..., 2] + K_rgb[1, 2]).astype(np.float32)
    P = np.stack([dirs[..., 0] * Z, dirs[..., 1] * Z, Z], -1)
    Q = P @ R.T + t
    u_d = (K_rgb[0, 0] * Q[..., 0] / np.maximum(Q[..., 2], 1e-6) + K_rgb[0, 2]).astype(np.float32)
    v_d = (K_rgb[1, 1] * Q[..., 1] / np.maximum(Q[..., 2], 1e-6) + K_rgb[1, 2]).astype(np.float32)
    return (u_inf, v_inf), (u_d, v_d)


def mi_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 32) -> float:
    va = np.clip(a[mask], 0, 1)
    vb = np.clip(b[mask], 0, 1)
    if va.size < 500:
        return float("nan")
    ia = np.clip((va * (bins - 1)).round().astype(np.int32), 0, bins - 1)
    ib = np.clip((vb * (bins - 1)).round().astype(np.int32), 0, bins - 1)
    H, _, _ = np.histogram2d(ia, ib, bins=bins, range=[[0, bins], [0, bins]])
    P = H / H.sum()
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    return float(np.sum(P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])))


def sf(g: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(g, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(g, axis=1) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def entropy(g: np.ndarray) -> float:
    h, _ = np.histogram(g, bins=256, range=(0, 1))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--tau", type=float, default=15.0)
    ap.add_argument("--taus", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0, 30.0])
    ap.add_argument("--geom_scale", type=float, default=16.0)
    ap.add_argument("--near_px", type=float, default=20.0)
    ap.add_argument("--far_px", type=float, default=5.0)
    ap.add_argument("--save_examples", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples").mkdir(exist_ok=True)

    seqs = sorted(p for p in args.data.iterdir() if p.is_dir())
    rows: list[dict] = []
    for seq in seqs:
        K_rgb, K_thr, M_t2r = load_calib(seq)
        rgb_files = sorted(next(seq.glob("sync_data/*/rgb/img_left")).glob("*.png"))
        thr_files = sorted(next(seq.glob("sync_data/*/thr/img_left")).glob("*.png"))
        dep_files = sorted(next(seq.glob("proj_depth/*/thr/depth_filtered")).glob("*.png"))
        idx = np.linspace(0, len(rgb_files) - 1,
                          min(args.max_frames, len(rgb_files))).astype(int)
        for j, i in enumerate(idx):
            rgb = cv2.imread(str(rgb_files[i]), cv2.IMREAD_COLOR)
            thr16 = cv2.imread(str(thr_files[i]), cv2.IMREAD_ANYDEPTH)
            dt = cv2.imread(str(dep_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
            Hh, Ww = thr16.shape
            thr = np.clip((thr16.astype(np.float32) - 3500.0) / (3969.0 - 3500.0), 0, 1)
            Z = dense_depth(dt)
            (u_i, v_i), (u_d, v_d) = mappings(K_rgb, K_thr, M_t2r, Z)
            d = np.hypot(u_d - u_i, v_d - v_i)

            raw = cv2.resize(rgb, (Ww, Hh), interpolation=cv2.INTER_LINEAR)
            H_im = cv2.remap(rgb, u_i, v_i, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            full = cv2.remap(rgb, u_d, v_d, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g_near = (d > args.tau).astype(np.float32)[..., None]
            g_soft = np.clip(1.0 - d / args.geom_scale, 0, 1)[..., None]
            Hf = H_im.astype(np.float32)
            conds = {
                "raw": raw.astype(np.float32),
                "H_dense": Hf,
                "H_gate_near": raw + g_near * (Hf - raw),
                "H_soft_far": raw + g_soft * (Hf - raw),
                "full": full.astype(np.float32),
            }
            for t in args.taus:
                conds[f"H_far@{t:g}"] = raw + (d < t).astype(np.float32)[..., None] * (Hf - raw)
            near = d > args.near_px
            far = d < args.far_px
            allm = np.ones_like(d, bool)
            allm[:8, :8] = False
            allm[-8:, -8:] = False

            for name, img in conds.items():
                g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                fused = 0.5 * (g + thr)
                rows.append(dict(
                    seq=seq.name, frame=int(i), cond=name,
                    mi_all=mi_masked(g, thr, allm),
                    mi_near=mi_masked(g, thr, near & allm),
                    mi_far=mi_masked(g, thr, far & allm),
                    sf=sf(fused), en=entropy(fused),
                    parallax_med=float(np.median(d)),
                    parallax_p95=float(np.percentile(d, 95)),
                    depth_valid=float((dt > 0).mean()),
                    gate_mean=float((d < float(name.split("@")[1])).mean()
                                    if name.startswith("H_far@")
                                    else (g_near.mean() if name == "H_gate_near"
                                          else (g_soft.mean() if name == "H_soft_far" else 1.0))),
                ))
            if j < args.save_examples:
                panel = np.vstack([
                    cv2.resize(rgb, (Ww, Hh)),
                    cv2.cvtColor((thr * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    np.clip(conds["raw"], 0, 255).astype(np.uint8),
                    np.clip(conds["H_dense"], 0, 255).astype(np.uint8),
                    np.clip(conds[f"H_far@{args.tau:g}"], 0, 255).astype(np.uint8),
                    np.clip(conds["full"], 0, 255).astype(np.uint8)])
                cv2.imwrite(str(args.out / "examples" / f"{seq.name}_{i:06d}.png"), panel)
            if (j + 1) % 25 == 0:
                print(f"  {seq.name}: {j+1}/{len(idx)}")

    conds = ["raw", "H_dense", "H_gate_near", "H_soft_far", "full"] + \
            [f"H_far@{t:g}" for t in args.taus]
    summary = {}
    for c in conds:
        sub = [r for r in rows if r["cond"] == c]
        summary[c] = {k: float(np.nanmean([r[k] for r in sub]))
                      for k in ("mi_all", "mi_near", "mi_far", "sf", "en", "gate_mean")}
    geo = {
        "parallax_med": float(np.mean([r["parallax_med"] for r in rows if r["cond"] == "raw"])),
        "parallax_p95": float(np.mean([r["parallax_p95"] for r in rows if r["cond"] == "raw"])),
        "depth_valid": float(np.mean([r["depth_valid"] for r in rows if r["cond"] == "raw"])),
    }
    out = {"n_frames": len([r for r in rows if r["cond"] == "raw"]),
           "tau": args.tau, "geom_scale": args.geom_scale,
           "near_px": args.near_px, "far_px": args.far_px,
           "geometry": geo, "conditions": summary}
    (args.out / "summary.json").write_text(json.dumps(out, indent=2))
    with (args.out / "summary.md").open("w") as f:
        f.write(f"# MS2 real-parallax diagnostic ({out['n_frames']} frames, "
                f"tau={args.tau} px, s_g={args.geom_scale} px)\n\n")
        f.write(f"- median parallax: {geo['parallax_med']:.2f} px "
                f"(p95 of the field {geo['parallax_p95']:.2f} px)\n")
        f.write(f"- thermal depth valid: {100*geo['depth_valid']:.1f}%\n\n")
        f.write("| condition | MI all | MI near (d>20) | MI far (d<5) | SF | EN | gate mean |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in conds:
            s = summary[c]
            f.write(f"| {c} | {s['mi_all']:.4f} | {s['mi_near']:.4f} | {s['mi_far']:.4f} | "
                    f"{s['sf']:.5f} | {s['en']:.4f} | {s['gate_mean']:.3f} |\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
