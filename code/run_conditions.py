#!/usr/bin/env python3
"""VTMOT detection-level experiment under real IR-VIS misalignment.

For each sequence a *global* affine (IR -> VIS) is estimated once from several
frames with ECC on gradient images, modelling a calibrated but parallax-blind rig.
Four composites are evaluated with the frozen M3FD detector against the MOT boxes
(pedestrian -> People, car -> Car):

  none      : 0.5 * (IR + VIS)
  H         : 0.5 * (warp(IR) + VIS), the global model applied densely
  gate      : none + N * (H - none), where N is a *locally observable* reliability
              map: per tile, the gradient-correlation gain of the warped over the
              unwarped IR against VIS, smoothed and clipped
  uniform   : N replaced by its mean (matched-activation control)

No labels are used to build N, so the gate is deployable; the detector evaluation
then measures whether spatial allocation of the global model helps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/sensors-4626757")


def read_gray(p: Path) -> np.ndarray:
    a = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)


def prep(a: np.ndarray, size=(320, 256)) -> np.ndarray:
    a = cv2.resize(a, size)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    g = np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0) + cv2.Sobel(a, cv2.CV_32F, 0, 1))
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return (g * 255).astype(np.uint8)


def estimate_affine(ir_paths: list[Path], vi_paths: list[Path], n: int = 8) -> tuple[np.ndarray, list[float]]:
    idx = np.linspace(0, len(ir_paths) - 1, min(n, len(ir_paths))).astype(int)
    ws, ccs = [], []
    for i in idx:
        ir, vg = read_gray(ir_paths[i]), read_gray(vi_paths[i])
        W = np.eye(2, 3, dtype=np.float32)
        try:
            cc, W = cv2.findTransformECC(
                prep(vg), prep(ir), W, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
        except cv2.error:
            continue
        if cc > 0.3:
            ws.append(W)
            ccs.append(cc)
    if not ws:
        return np.eye(2, 3, dtype=np.float32), ccs
    return np.median(np.stack(ws), 0).astype(np.float32), ccs


def gradient(a: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def tile_gain(a: np.ndarray, b: np.ndarray, h: int, w: int, win: int = 31, stride: int = 8) -> np.ndarray:
    """Per-tile gradient correlation between a and b, returned on the tile grid."""
    ga, gb = gradient(a), gradient(b)
    ys = np.arange(0, h - win + 1, stride)
    xs = np.arange(0, w - win + 1, stride)
    out = np.zeros((len(ys), len(xs)), np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            A = ga[y:y + win, x:x + win].ravel()
            B = gb[y:y + win, x:x + win].ravel()
            if A.std() < 1e-3 or B.std() < 1e-3:
                out[iy, ix] = 0.0
                continue
            A = (A - A.mean()) / A.std()
            B = (B - B.mean()) / B.std()
            out[iy, ix] = float((A * B).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, default=BASE / "exp/vtmot/subset")
    ap.add_argument("--out", type=Path, default=BASE / "exp/vtmot/run")
    ap.add_argument("--yolo", default=str(BASE.parent / "eval/m3fd_yolov8n.pt"))
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--protocol", choices=["m3fd", "coco"], default="m3fd",
                    help="label space of the evaluated detector")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.subset / "meta.json").read_text())
    conds = ["none", "H", "gate", "uniform"]
    for c in conds:
        (args.out / c).mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(exist_ok=True)

    info = {}
    for seq in meta["sequences"]:
        ir_paths = sorted((args.subset / seq / "infrared").glob("*.jpg"))
        vi_paths = sorted((args.subset / seq / "visible").glob("*.jpg"))
        W, ccs = estimate_affine(ir_paths, vi_paths)
        Wf = W.copy()
        Wf[0, 2] *= 2.0
        Wf[1, 2] *= 2.0
        info[seq] = {"affine_320": W.tolist(), "affine_640": Wf.tolist(),
                     "ecc_cc": [round(c, 3) for c in ccs]}
        print(f"{seq}: ecc frames={len(ccs)} cc_med={np.median(ccs):.3f} "
              f"disp={2*np.hypot(*(W@np.array([[0,0,1],[319,0,1],[0,255,1],[319,255,1]],np.float32).T)[:2]-np.array([[0,0],[319,0],[0,255],[319,255]],np.float32).T ).mean():.1f}px")

        for ir_p in ir_paths:
            name = f"{seq}_{ir_p.stem}"
            vi_p = args.subset / seq / "visible" / f"{ir_p.stem}.jpg"
            ir = read_gray(ir_p)
            vi = cv2.cvtColor(cv2.imread(str(vi_p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            h, w = vi.shape
            warped = cv2.warpAffine(ir, Wf, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE)
            none = 0.5 * (ir.astype(np.float32) + vi.astype(np.float32))
            H = 0.5 * (warped.astype(np.float32) + vi.astype(np.float32))

            g_raw = tile_gain(ir, vi, h, w)
            g_warp = tile_gain(warped, vi, h, w)
            gain = np.clip(g_warp - g_raw, 0, None)          # observable gain
            gain = gain / (gain.max() + 1e-6)
            gy, gx = gain.shape
            gate_small = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
            gate = np.clip(cv2.GaussianBlur(gate_small, (0, 0), 9), 0, 1)
            uni = np.full_like(gate, float(gate.mean()))

            outs = {
                "none": none,
                "H": H,
                "gate": none + gate * (H - none),
                "uniform": none + uni * (H - none),
            }
            for c, img in outs.items():
                u8 = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(args.out / c / f"{name}.png"),
                            cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR))
            lab_src = args.subset / "labels" / f"{name}.txt"
            if lab_src.exists():
                txt = lab_src.read_text()
                if args.protocol == "coco":          # People->person(0), Car->car(2)
                    out = []
                    for ln in txt.splitlines():
                        parts = ln.split()
                        if len(parts) == 5:
                            parts[0] = "0" if parts[0] == "0" else "2"
                            out.append(" ".join(parts))
                    txt = "\n".join(out) + "\n"
                (args.out / "labels" / f"{name}.txt").write_text(txt)

    (args.out / "registration.json").write_text(json.dumps(info, indent=2))

    if args.eval:
        import tempfile, shutil, yaml
        from ultralytics import YOLO
        res = {}
        for c in conds:
            tmp = Path(tempfile.mkdtemp(prefix=f"vtmot_{c}_"))
            (tmp / "images/val").mkdir(parents=True)
            (tmp / "labels/val").mkdir(parents=True)
            for img in sorted((args.out / c).glob("*.png")):
                (tmp / "images/val" / img.name).symlink_to(img.resolve())
                lab = args.out / "labels" / f"{img.stem}.txt"
                if lab.exists():
                    (tmp / "labels/val" / f"{img.stem}.txt").symlink_to(lab.resolve())
            if args.protocol == "coco":
                cfg = {"path": str(tmp), "train": "images/val", "val": "images/val", "nc": 80,
                       "names": [f"c{i}" for i in range(80)]}
            else:
                cfg = {"path": str(tmp), "train": "images/val", "val": "images/val",
                       "nc": 6, "names": ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]}
            (tmp / "data.yaml").write_text(yaml.dump(cfg))
            model = YOLO(args.yolo)
            m = model.val(data=str(tmp / "data.yaml"), imgsz=640, batch=8,
                          device="cuda", classes=args.classes, verbose=False, plots=False)
            res[c] = {"mAP50": round(float(m.box.map50), 4),
                      "mAP50_95": round(float(m.box.map), 4),
                      "P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4),
                      "n_images": len(list((args.out / c).glob("*.png")))}
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  {c}: mAP50={res[c]['mAP50']:.4f} (P={res[c]['P']:.3f} R={res[c]['R']:.3f})")
        (args.out / "detection_results.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
