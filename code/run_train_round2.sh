#!/usr/bin/env bash
# Round-2 retraining driver: 3 seeds x {product, task}, two batches.
# Batch A: product (end-to-end necessity), 30 epochs.
# Batch B: task (task-importance), 25 epochs.
# Logs: sensors-4626757/exp/logs/train_*.log
set -u
REPO=/home/ubuntu/Code/Registration-Necessity-Guided-tsdr_register
cd "$REPO"
export PYTHONPATH="$REPO"
PY=/home/ubuntu/miniconda3/envs/tdsr/bin/python
R="$REPO/过程文件/revision_2026-09"
DATA=/home/ubuntu/Code/datasets/M3FD_rebuilt
IMP="$R/cache/task_importance/M3FD"
LOGS="$REPO/sensors-4626757/exp/logs"
mkdir -p "$LOGS" "$R/results"

run_one () {  # target seed epochs
  local target=$1 seed=$2 epochs=$3
  local name
  if [ "$target" = "product" ]; then name="necessity_seed$seed"; else name="task_importance_seed$seed"; fi
  "$PY" "$R/scripts/train_necessity.py" \
    --target "$target" --seed "$seed" --epochs "$epochs" \
    --data_root "$DATA" --imp_root "$IMP" --output "$R/results/$name" \
    > "$LOGS/train_${target}_seed${seed}.log" 2>&1
  echo "[done] $target seed=$seed -> $R/results/$name (rc=$?)"
}

echo "=== batch A: product x3 ==="
for s in 0 1 2; do run_one product "$s" 30 & done
wait
echo "=== batch B: task x3 ==="
for s in 0 1 2; do run_one task "$s" 25 & done
wait
echo "=== summary ==="
for d in "$R"/results/necessity_seed* "$R"/results/task_importance_seed*; do
  [ -f "$d/summary.json" ] && echo "$(basename "$d") $(grep -o '"best_val_spearman":[^,]*' "$d/summary.json")"
done
