#!/usr/bin/env bash
#SBATCH --job-name=prc-a2cv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-179%6
#
# A3 ARM 2 (ADS-B + wake + backlog + gap cap, cluster/w2_extra.py): campaign, CPU members: the W2 families + the adsb.lol columns (cluster/w2_task.py --adsb),
# 5 families x 12 months x 3 seeds = 180 fold fits into cache/cv/adsb2/.
# Index order is SEED-MAJOR, so tasks 0-59 are seed 0 of every family and month: after
# they finish, experiments/w2/w2_eval.py can already compare seed 0 against seed 0.
#
#   sbatch -p gpu --dependency=afterok:<stage-2 job> cluster/adsb2_cv_array.sbatch            (seed 0 = --array=0-59%6)
#
# Partition: -p gpu runs CPU-only jobs fine without --gres and was idle on 2026-09-24.
# %6 x 8 CPUs = 48 of its 64 cores, leaving 16 for the network array (w2_nn_array.sbatch).
# Each fit is expected at ~5-15 min (study fits took 3-4.5 min with 72 columns; W2 has 96),
# so `-p any --time=00:59:00` would also work -- `any` has MaxTime 01:00:00 and anything
# longer sits PENDING with (PartitionTimeLimit).
#
# Inputs on the cluster: cache/training.parquet, cache/external/ (already there) and
# cache/v11/feat_training.parquet (rsync from the laptop). Idempotent: a fold whose output
# exists in cache/cv/adsb2/ is skipped, so resubmit failed indices with --array=<list>.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet; do [ -f "$f" ] || { echo "FATAL: $f missing"; exit 2; }; done

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

srun python cluster/w2_task.py --family "$FAM" --seed "$SEED" --month "$MONTH" --adsb --extra --campaign adsb2 \
     --threads "$SLURM_CPUS_PER_TASK"
