#!/usr/bin/env bash
#SBATCH --job-name=prc-famfinal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-11%4
#
# Final fits for the v10 family blend: 4 families x 3 seeds = 12 tasks.
#
#   sbatch cluster/family_final.sbatch                  # all 12 on the GPU node's CPUs, 4 at a time
#   sbatch --array=0-8 cluster/family_final.sbatch      # perapt/lintree/xgboost only (v12 needs no routed)
#
# WHY -p gpu (no GPU requested): on 2026-09-24 the same array on `any` sat PENDING with
# reason PartitionTimeLimit -- `any` does not allow the 3 h limit, so it could never start.
# The GPU node allows up to 7 days and its 64 CPUs were idle; 4 tasks x 8 CPUs = 32 cores.
#
#   index: family = (perapt lintree xgboost routed)[i / 3], seed = i % 3
#
# 3 h rather than the study's 1 h: fitting on all twelve months is ~9% more data
# and catboost showed how easily a 1 h limit wastes whole tasks. Writes
# cache/family_final/<family>__seedNN.npy + ranking_ids.npy; existing files are
# skipped. Pull them back with
#   rsync -az vivaldi:03_2026_PRC_Challenge/2026-PRC-Challenge/cache/family_final/ \
#         <laptop repo>/cache/family_final/
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

FAMS=(perapt lintree xgboost routed)
i=$SLURM_ARRAY_TASK_ID
srun python cluster/family_final.py --family "${FAMS[$(( i / 3 ))]}" --seed "$(( i % 3 ))" \
     --threads "$SLURM_CPUS_PER_TASK"
