#!/usr/bin/env bash
#SBATCH --job-name=prc-c1cv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-179%6
#
# C1 CONSOLIDATED RETRAIN (2026-09-26), CPU members: the A3 arm-2 columns (W2 + adsb.lol + wake / backlog / gap cap)
# + the 8 de-icing / disruption share columns (cluster/w2_features.py DEICE_COLS), into cache/cv/c1/.
# 5 families x 12 months x 3 seeds = 180 fold fits, SEED-MAJOR (0-59 = seed 0 of every family and month).
#
#   sbatch cluster/c1_cv_array.sbatch
#
# -p gpu without --gres runs CPU-only; %6 x 8 CPUs = 48 of 64 cores, leaving 16 for c1_nn_array.sbatch.
# The adsb2 folds took ~6-7 min each. Idempotent: a fold whose output exists in cache/cv/c1/ is
# skipped, so resubmit failed indices with --array=<list>. cache/cv/c1/_config.json pins the flags.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet cache/deice_0926/ind_training.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first"; exit 2; }
done

FAMS=(mix perapt lintree xgboost routed)
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 60 ))
REM=$(( i % 60 ))
FAM=${FAMS[$(( REM / 12 ))]}
MONTH=$(( REM % 12 + 1 ))
echo "task $i -> family $FAM seed $SEED month $MONTH"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026          # the CPU env: has xgboost (prc2026gpu does not)

srun python cluster/w2_task.py --family "$FAM" --seed "$SEED" --month "$MONTH" \
     --adsb --extra --deice --nn-missing sentinel --campaign c1 --threads "$SLURM_CPUS_PER_TASK"
