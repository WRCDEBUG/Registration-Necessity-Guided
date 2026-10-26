# Release — registration necessity / selective correction study

Public repository: https://github.com/WRCDEBUG/Registration-Necessity-Guided

This bundle accompanies the revision of the manuscript. It contains the code, the
checkpoints and the per-image predictions needed to regenerate every number in the
paper and in the response letter. `results/numbers.csv` is the single source of
truth: every reported value has an entry with the path of the artefact that
produced it.

## Contents

```
code/         experiment scripts (gates, registration, fusion metrics, table/revision builders)
checkpoints/  3 seeds x {product, task} for M3FD, 1 in-domain LLVIP task model,
              and the recovered proxy-supervised model of the previous submission
manifests/    M3FD test split (1049 ids), baseline cache manifest, seed and fairness tables
results/      detection_results.json and per-image CSVs for every experiment,
              plus numbers.csv
logs/         run logs (empty in the bundle; regenerate with the commands below)
SHA256SUMS    hashes of everything above
```

## Environment

Python 3.10, PyTorch 2.3.1+cu121, torchvision 0.18.1, ultralytics 8.4.87,
OpenCV 5.0.0, scipy 1.15, timm 1.0.27. One A100 (80 GB) was used for all runs.

## What each table needs

| paper artefact | script | output |
|---|---|---|
| Table 3 (gate matrix) | `code/eval_gates_ext.py` | `results/e1_seed*__detection.json`, `results/geom_rules__detection.json` |
| Table 4 (seeds) | `results/e1_seed*__detection.json` | `manifests/seed_table.csv` |
| Table 5 (gamma) | `code/eval_gates_ext.py --gammas 0.25 0.5 0.75 1` | `results/gamma_new__detection.json` |
| Table 6 (real registration) | `code/est_registration_v2.py` then `code/eval_est_reg.py` | `results/eval_real__detection.json` |
| Table 7 (error sweep) | `code/eval_sim_reg.py --err_px <e>` | `results/sim_e*__detection.json` |
| Table 8 (magnitude) | `code/eval_gates_ext.py --angle_range/--translate_range/--scale_range` | `results/trans*__detection.json` |
| Table 9 (fusion quality) | `code/fusion_metrics.py --ref_mode clean` | `results/fusion_quality_clean.csv` |
| Table 10 (LLVIP) | `code/eval_gates_ext.py` with the LLVIP subset and task checkpoint | `results/full_decomp__detection.json` |
| Table 11 (equal information access) | `manifests/fairness_table.csv` + published baseline outputs | `manifests/fairness_table.csv` |
| all tables as LaTeX | `code/make_tables.py` | `tables/*.tex` |

## Third-party components

* MI / Qabf / VIF / SSIM use the implementations of the VF-Bench metric suite
  (https://github.com/VideoFusion), used unmodified; EN / SD / SF / SCD are
  implemented in `code/fusion_metrics.py`.
* Detection uses Ultralytics YOLOv8n and RT-DETR-L. The M3FD detector was
  trained on grayscale fused images of the 3151 training pairs and validated on
  the M3FD infrared test split; the LLVIP detector is trained on LLVIP infrared.
  Both configurations are reported in the manuscript and in `AUDIT.md`.
* Datasets: M3FD (TarDAL) and LLVIP, publicly available from their authors.

## Known limitations (also stated in the manuscript)

1. The archived LLVIP absolute numbers of the previous submission cannot be
   reproduced with the documented detector; the ordering and the gaps are.
2. The M3FD detector was selected on the evaluation split and trained on fused
   images produced by this project; both facts bias absolute scores and are
   disclosed in §4.1.3 and §4.8 of the revision.
3. The estimated registration module is one representative implementation of
   classic cross-modal registration (median corner error 8.1 px).
4. The error sweep assumes a smooth error field uncorrelated with image content.

## Licence

Code is released for review and reuse with attribution. Checkpoints inherit the
licences of the models they were fine-tuned from (Ultralytics YOLOv8, RT-DETR).
Dataset images are not redistributed; only file lists and derived prediction
artefacts are included.
