#!/usr/bin/env bash
# Table 7 (misalignment sweep) and Table 8 (input noise) with the retrained task model,
# 300-pair deterministic subset, plus per-image analysis for paired CIs.
set -u
REPO=/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register
cd "$REPO"; export PYTHONPATH="$REPO"
PY=/home/ubuntu/miniconda3/envs/tdsr/bin/python
R="$REPO/过程文件/revision_2026-09"
PYT="$REPO/sensors-4626757/exp/gates/eval_gates_ext.py"
ANA="$R/scripts/analyse_detections.py"
YOLO="$REPO/eval/m3fd_yolov8n.pt"
LBL=/home/ubuntu/Code/datasets/M3FD_rebuilt/labels/test
OUT="$REPO/sensors-4626757/exp/sweep"
LOGS="$REPO/sensors-4626757/exp/logs"
run () {  # tag angle trans scale noise
  local tag=$1 ang=$2 tr=$3 sc=$4 nz=$5
  "$PY" -u "$PYT" --yolo "$YOLO" --task_ckpt "$R/results/task_importance_seed0/best.pt" \
    --angle_range "$ang" --translate_range "$tr" --scale_range "$sc" --noise_sigma "$nz" \
    --gates learned_factorized,uniform --gammas 1.0 --max_samples 300 \
    --output "$OUT/$tag" > "$LOGS/sweep_$tag.log" 2>&1
  "$PY" -u "$ANA" --images_root "$OUT/$tag/images" --label_dir "$LBL" --yolo "$YOLO" \
    --conditions none,full,learned_factorized_g1,uniform_g1 --baseline none \
    --output "$OUT/$tag/per_image" > "$LOGS/sweep_${tag}_ana.log" 2>&1
  echo "[sweep] $tag done"
}
run trans2   0 2  0    0
run trans8   0 8  0    0
run trans16  0 16 0    0
run trans24  0 24 0    0
run rot0     0 8  0    0
run rot8     8 8  0    0
run noscale  3 8  0    0
run scale8   3 8  0.08 0
run noise003 3 8  0.03 0.03
run noise008 3 8  0.03 0.08
echo "=== t7/t8 done ==="
