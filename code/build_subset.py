#!/usr/bin/env python3
"""Build a VTMOT detection subset with real IR-VIS misalignment.

VTMOT (VT-MOT) ships paired thermal/visible UAV sequences at 640x512 with MOT
annotations, so it provides detection labels *and* a real, non-synthetic
cross-modal misalignment. Classes are 1 = pedestrian, 2 = car; they are mapped to
the M3FD detector's label space (0 = People, 1 = Car).

Output layout (under --out):
  <seq>/infrared/<frame>.jpg, <seq>/visible/<frame>.jpg
  labels/<seq>_<frame>.txt          YOLO boxes for People/Car
  meta.json                         frames, sequences, box counts
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

ZIP = Path("/home/ubuntu/Code/datasets/VYMOT/VTMOT_test.zip")
ROOT = "data1/Datasets/Tracking/MOT/VTMOT/images/test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register/"
                                 "sensors-4626757/exp/vtmot/subset"))
    ap.add_argument("--sequences", nargs="+",
                    default=["wurenji-0303-17", "wurenji-0303-09", "Vtuav-02",
                             "LasHeR-096", "wurenji-0305-13", "LasHeR-004"])
    ap.add_argument("--frames_per_seq", type=int, default=40)
    ap.add_argument("--min_boxes", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(exist_ok=True)

    zf = zipfile.ZipFile(ZIP)
    names = set(zf.namelist())
    meta = {"sequences": {}, "class_map": {"1": 0, "2": 1},
            "class_names": {"0": "People", "1": "Car"}}

    for seq in args.sequences:
        gt_path = f"{ROOT}/{seq}/gt/gt.txt"
        if gt_path not in names:
            print("skip (no gt):", seq)
            continue
        rows = [ln.split(",") for ln in zf.read(gt_path).decode("utf-8", "ignore").splitlines()
                if len(ln.split(",")) >= 8]
        by_frame: dict[int, list[list[str]]] = {}
        for r in rows:
            if r[7] not in ("1", "2"):
                continue
            by_frame.setdefault(int(float(r[0])), []).append(r)
        usable = sorted(f for f, v in by_frame.items() if len(v) >= args.min_boxes)
        if not usable:
            print("skip (no usable frames):", seq)
            continue
        pick = np.linspace(0, len(usable) - 1, args.frames_per_seq).round().astype(int)
        frames = [usable[i] for i in sorted(set(pick.tolist()))]
        (args.out / seq / "infrared").mkdir(parents=True, exist_ok=True)
        (args.out / seq / "visible").mkdir(parents=True, exist_ok=True)
        for f in frames:
            for kind in ("infrared", "visible"):
                src = f"{ROOT}/{seq}/{kind}/{f:06d}.jpg"
                if src not in names:
                    print("missing", src)
                    continue
                (args.out / seq / kind).parent.mkdir(parents=True, exist_ok=True)
                (args.out / seq / kind / f"{f:06d}.jpg").write_bytes(zf.read(src))
            with (args.out / "labels" / f"{seq}_{f:06d}.txt").open("w") as fh:
                for r in by_frame[f]:
                    x, y, w, h = map(float, r[2:6])
                    cls = 0 if r[7] == "1" else 1
                    xc, yc = (x + w / 2) / 640.0, (y + h / 2) / 512.0
                    fh.write(f"{cls} {xc:.6f} {yc:.6f} {w/640.0:.6f} {h/512.0:.6f}\n")
        meta["sequences"][seq] = {
            "frames": frames, "n_frames": len(frames),
            "n_boxes": int(sum(len(by_frame[f]) for f in frames)),
            "n_people": int(sum(1 for f in frames for r in by_frame[f] if r[7] == "1")),
            "n_car": int(sum(1 for f in frames for r in by_frame[f] if r[7] == "2")),
        }
        print(f"{seq}: {len(frames)} frames, {meta['sequences'][seq]['n_boxes']} boxes "
              f"(people {meta['sequences'][seq]['n_people']}, car {meta['sequences'][seq]['n_car']})")

    meta["total_frames"] = sum(v["n_frames"] for v in meta["sequences"].values())
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("total frames:", meta["total_frames"])


if __name__ == "__main__":
    main()
