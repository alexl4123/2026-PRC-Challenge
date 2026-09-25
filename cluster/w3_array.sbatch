#!/usr/bin/env bash
#SBATCH --job-name=prc-w3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%4
#
# W3, the second network architecture (prc/models/w3_net.py): 12 months x 3 seeds = 36 fold fits
# (cluster/w2_task.py --family w3), W2 feature set, written into cache/cv/w2/w3__seedNN__mMM.npy.
# Index order is SEED-MAJOR (0-11 = seed 0).
#
#   sbatch cluster/w3_array.sbatch
#
# --gres=gpu:1 is MANDATORY on -p gpu (devices are confined per cgroup); w2_task.py stops
# at once if no GPU is visible. %4 x 4 CPUs = 16 cores, which together with w2_array.sbatch's
# 48 fills the node's 64. --mem 40G: the study's network tasks used 32G with 72 columns.
# Idempotent, like w2_array.sbatch.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
[ -f cache/v11/feat_training.parquet ] || { echo "FATAL: cache/v11/feat_training.parquet missing -- rsync it first"; exit 2; }

i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 12 ))
MONTH=$(( i % 12 + 1 ))
echo "task $i -> w3 seed $SEED month $MONTH"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

srun python cluster/w2_task.py --family w3 --seed "$SEED" --month "$MONTH" --device cuda \
     --threads "$SLURM_CPUS_PER_TASK"
