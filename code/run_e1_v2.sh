#!/usr/bin/env bash
# E1 v2: tau selected on a held-out val300 subset (train split), then evaluated on test.
set -u
REPO=/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register
cd "$REPO"; export PYTHONPATH="$REPO"
PY=/home/ubuntu/miniconda3/envs/tdsr/bin/python
R="$REPO/过程文件/revision_2026-09"
PYT="$REPO/sensors-4626757/exp/gates/eval_gates_ext.py"
LOGS="$REPO/sensors-4626757/exp/logs"; OUT="$REPO/sensors-4626757/exp/gates"
YOLO="$REPO/eval/m3fd_yolov8n.pt"; VAL="$REPO/sensors-4626757/exp/val300"
LEGACY="$REPO/sensors-4626757/audit/recovered/tsdr_r1/results_full_proxy_mult_beta05_v2/checkpoints/stage3_best_spearman.pth"

tau_val () {   # 5 thresholds on val300 (300 pairs, ~2 min each)
  for t in 0.1 0.2 0.3 0.5 0.7; do
    "$PY" "$PYT" --yolo "$YOLO" --data_root "$VAL" --split val \
      --label_dir "$VAL/labels/val" --gates geom_thresh --tau "$t" --gammas 1.0 \
      --output "$OUT/tauval_$t" > "$LOGS/tauval_$t.log" 2>&1
    echo "[tauval] $t done"
  done
}
seed_full () {  # seed 0, full gate set on test
  "$PY" "$PYT" --yolo "$YOLO" \
    --ckpt "$R/results/necessity_seed0/best.pt" \
    --task_ckpt "$R/results/task_importance_seed0/best.pt" \
    --legacy_ckpt "$LEGACY" \
    --gates none,full,oracle_geom,oracle_task,oracle_product,uniform,shuffled,learned,learned_factorized,task_only,abs,proxy_net \
    --gammas 1.0 --output "$OUT/e1_seed0" > "$LOGS/e1_seed0.log" 2>&1
  echo "[e1] seed0 done"
}
seed_lean () {  # seeds 1,2: only the two rows that need a seed
  local s=$1
  "$PY" "$PYT" --yolo "$YOLO" \
    --ckpt "$R/results/necessity_seed$s/best.pt" \
    --task_ckpt "$R/results/task_importance_seed$s/best.pt" \
    --gates none,full,learned,learned_factorized,task_only \
    --gammas 1.0 --output "$OUT/e1_seed$s" > "$LOGS/e1_seed$s.log" 2>&1
  echo "[e1] seed$s done"
}
tau_val & seed_full & seed_lean 1 & seed_lean 2 &
wait
echo "=== all done ==="
