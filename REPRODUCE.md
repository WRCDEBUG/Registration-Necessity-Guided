# Reproduction commands

All commands assume the repository layout of the manuscript project and the
public M3FD/LLVIP datasets. `$P` is the project root, `$S = $P/sensors-4626757`.

```bash
export PYTHONPATH=$P
PY=python   # environment: torch 2.3.1+cu121, ultralytics 8.4.87, opencv 5.0.0

# 0. task-importance cache (M3FD + LLVIP)
$PY $P/过程文件/revision_2026-09/scripts/build_task_importance.py \
  --m3fd  $P/../datasets/M3FD_rebuilt --llvip $P/../datasets/LLVIP \
  --out   $P/过程文件/revision_2026-09/cache/task_importance

# 1. train the six M3FD models (3 seeds x 2 targets)
for s in 0 1 2; do
  $PY .../train_necessity.py --target product --seed $s --epochs 30 --output .../necessity_seed$s
  $PY .../train_necessity.py --target task    --seed $s --epochs 25 --output .../task_importance_seed$s
done

# 2. gate matrix (Table 3) + seeds (Table 4)
for s in 0 1 2; do
  $PY $S/exp/gates/eval_gates_ext.py --yolo $P/eval/m3fd_yolov8n.pt \
    --ckpt .../necessity_seed$s/best.pt --task_ckpt .../task_importance_seed$s/best.pt \
    --gates none,full,oracle_geom,oracle_task,oracle_product,uniform,shuffled,learned,learned_factorized,task_only,abs \
    --gammas 1.0 --output $S/exp/gates/e1_seed$s
done

# 3. real registration (Tables 6) and error sweep (Table 7)
$PY $S/exp/registration/est_registration_v2.py --out $S/exp/registration/est_v2_test.npz
$PY $S/exp/registration/eval_est_reg.py --est $S/exp/registration/est_v2_test.npz \
  --output $S/exp/registration/eval_real
for e in 0.5 1.5 4.0 8.0 12.0; do
  $PY $S/exp/registration/eval_sim_reg.py --err_px $e --output $S/exp/registration/sim_e$e
done

# 4. fusion-quality metrics (Table 9)
$PY $S/exp/fusion/fusion_metrics.py --ref_mode clean --max_samples 300

# 5. rebuild every LaTeX table and the manuscript
$PY $S/paper/make_tables.py
$PY $S/paper/make_revision.py
cd $S && pdflatex main_revision_A && bibtex main_revision_A && pdflatex main_revision_A && pdflatex main_revision_A
```
