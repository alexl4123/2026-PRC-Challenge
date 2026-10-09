#!/usr/bin/env bash
#SBATCH --job-name=prc-nnfinal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# Final entity-embedding network: all of 2025 -> ranking set, one seed per task.
#
#   sbatch --array=0-2 cluster/nn_final.sbatch
#
# Needs cache/training.parquet, cache/ranking.parquet and cache/external/ on the
# cluster (already there from earlier data syncs; `bash cluster/sync.sh` re-sends
# them if not). The code itself arrives by `git pull`. Writes
# cache/nn_final/nn_final_seedNN.npy + ranking_ids.npy. Pull them back with
#   rsync -az cluster:<repo>/cache/nn_final/ \
#         <laptop repo>/cache/nn_final/
#
# --gres=gpu:1 is MANDATORY on -p gpu (devices are confined per cgroup; see
# cluster/nn_array.sbatch). --mem is 40G rather than 32G because the ranking
# rows are encoded alongside the training rows.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# final phase (2026-10-09): sbatch --export=ALL,PRC_RANKING=final ... reads/writes cache_final/
C=cache
if [ "${PRC_RANKING:-}" = final ]; then C=cache_final; fi
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

srun python cluster/nn_final.py --seed "$SLURM_ARRAY_TASK_ID" --device cuda \
     --threads "$SLURM_CPUS_PER_TASK"
