#!/usr/bin/env bash
# E6: phi / s_g sensitivity on the held-out val300 subset (no test-set tuning)
set -u
REPO=/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register
cd "$REPO"; export PYTHONPATH="$REPO"
PY=/home/ubuntu/miniconda3/envs/tdsr/bin/python
R="$REPO/过程文件/revision_2026-09"; PYT="$REPO/sensors-4626757/exp/gates/eval_gates_ext.py"
LOGS="$REPO/sensors-4626757/exp/logs"; OUT="$REPO/sensors-4626757/exp/sensitivity"
YOLO="$REPO/eval/m3fd_yolov8n.pt"; VAL="$REPO/sensors-4626757/exp/val300"
for phi in 0.0 0.25 0.5 0.75 1.0; do
  "$PY" "$PYT" --yolo "$YOLO" --data_root "$VAL" --split val --label_dir "$VAL/labels/val" \
    --task_ckpt "$R/results/task_importance_seed0/best.pt" --floor "$phi" --geom_scale 16 \
    --gates learned_factorized,task_only --gammas 1.0 --output "$OUT/phi_$phi" \
    > "$LOGS/e6_phi_$phi.log" 2>&1
  echo "[e6] phi=$phi done"
done
for sg in 8 32 64; do
  "$PY" "$PYT" --yolo "$YOLO" --data_root "$VAL" --split val --label_dir "$VAL/labels/val" \
    --task_ckpt "$R/results/task_importance_seed0/best.pt" --floor 0.5 --geom_scale "$sg" \
    --gates learned_factorized --gammas 1.0 --output "$OUT/sg_$sg" \
    > "$LOGS/e6_sg_$sg.log" 2>&1
  echo "[e6] s_g=$sg done"
done
echo "=== e6 done ==="
