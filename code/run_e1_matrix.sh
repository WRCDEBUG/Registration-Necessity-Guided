#!/usr/bin/env bash
# E1 extended gate matrix: tau sweep + 3-seed learned gates.
# Uses sensors-4626757/exp/gates/eval_gates_ext.py (numerically identical to the
# original evaluator for shared gates; validated on 40 samples).
set -u
REPO=/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register
cd "$REPO"
export PYTHONPATH="$REPO"
PY=/home/ubuntu/miniconda3/envs/tdsr/bin/python
R="$REPO/过程文件/revision_2026-09"
PYT="$REPO/sensors-4626757/exp/gates/eval_gates_ext.py"
LOGS="$REPO/sensors-4626757/exp/logs"
OUT="$REPO/sensors-4626757/exp/gates"
YOLO="$REPO/eval/m3fd_yolov8n.pt"
LEGACY="$REPO/sensors-4626757/audit/recovered/tsdr_r1/results_full_proxy_mult_beta05_v2/checkpoints/stage3_best_spearman.pth"

# ---- stream 1: geometry threshold sweep (seed independent) ----
tau_sweep () {
  for t in 0.1 0.2 0.3 0.5 0.7; do
    "$PY" "$PYT" --yolo "$YOLO" --gates geom_thresh --tau "$t" --gammas 1.0 \
      --output "$OUT/tau_$t" > "$LOGS/gates_tau_$t.log" 2>&1
    echo "[tau] $t done"
  done
}

# ---- stream 2: 3 seeds x learned gates ----
seed_run () {
  local s=$1
  "$PY" "$PYT" --yolo "$YOLO" \
    --ckpt "$R/results/necessity_seed$s/best.pt" \
    --task_ckpt "$R/results/task_importance_seed$s/best.pt" \
    --legacy_ckpt "$LEGACY" \
    --gates none,full,oracle_geom,oracle_task,oracle_product,uniform,shuffled,learned,learned_factorized,task_only,abs,proxy_net \
    --gammas 1.0 --output "$OUT/e1_seed$s" > "$LOGS/gates_e1_seed$s.log" 2>&1
  echo "[e1] seed $s done"
}

tau_sweep &
seed_run 0 &
wait
echo "=== stream 1+seed0 done; launching seeds 1,2 ==="
seed_run 1 & seed_run 2 &
wait
echo "=== all done ==="
