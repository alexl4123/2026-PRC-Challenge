#!/usr/bin/env bash
#SBATCH --job-name=prc-c1nn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-71%4
#
# C1 CONSOLIDATED RETRAIN (2026-09-26), network members: nn (embed_nn, cap 16, NaN -> -10 SENTINEL instead of
# the fold median) and w3 (prc/models/w3_net.py, unchanged; it has its own missing indicator), on the A3 arm-2
# columns + the 8 de-icing / disruption share columns, into cache/cv/c1/.
# 2 families x 12 months x 3 seeds = 72 fold fits, SEED-MAJOR: 0-23 = seed 0 (0-11 nn, 12-23 w3), 24-47 seed 1, 48-71 seed 2.
#
#   sbatch cluster/c1_nn_array.sbatch
#
# --gres=gpu:1 is MANDATORY on -p gpu. Idempotent: finished folds in cache/cv/c1/ are skipped.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet cache/deice_0926/ind_training.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first"; exit 2; }
done
FAMS=(nn w3)
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 24 ))
REM=$(( i % 24 ))
FAM=${FAMS[$(( REM / 12 ))]}
MONTH=$(( REM % 12 + 1 ))
echo "task $i -> $FAM seed $SEED month $MONTH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/w2_task.py --family "$FAM" --seed "$SEED" --month "$MONTH" --device cuda \
     --adsb --extra --deice --nn-missing sentinel --campaign c1 --threads "$SLURM_CPUS_PER_TASK"
