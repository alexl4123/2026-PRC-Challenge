#!/usr/bin/env bash
#SBATCH --job-name=prc-r1ret
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-3
#
# R1: TabICL in the bulk with retrieval context, one (airport, month) cell per task
# (experiments/tfm/r1_bulk_retrieval.py).
#
#   sbatch cluster/r1_array.sbatch
#
# -p gpu, not `any` (MaxTime=01:00:00 there). --gres=gpu:1 is MANDATORY on -p gpu.
# Inputs (rsync from the laptop, see claude_log/2026-09-24_dev.log §17):
#   cache/altfam_X.parquet, cache/altfam_y.npy, cache/altfam_month.npy, cache/strategy/v11s_oof.npy
# and the TabICL regressor checkpoint in ~/.cache/huggingface/hub/models--jingang--TabICL.
# Idempotent: a finished cell's npz in cache/tfm_r1/ is skipped. Pull back and report on the laptop:
#   rsync -az vivaldi:03_2026_PRC_Challenge/2026-PRC-Challenge/cache/tfm_r1/ <laptop repo>/cache/tfm_r1/
#   python experiments/tfm/r1_bulk_retrieval.py --report
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

for f in cache/altfam_X.parquet cache/altfam_y.npy cache/altfam_month.npy cache/strategy/v11s_oof.npy; do
    [ -f "$f" ] || { echo "FATAL: missing input $f -- rsync it from the laptop first"; exit 2; }
done
ls "$HOME"/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/tabicl-regressor-v2-20260212.ckpt >/dev/null \
    || { echo "FATAL: TabICL regressor checkpoint not in ~/.cache/huggingface/hub -- rsync it first"; exit 2; }

CELLS=(LIRF:7 EDDF:1 EHAM:7 LFPG:1)
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
export HF_HUB_OFFLINE=1

srun python experiments/tfm/r1_bulk_retrieval.py --device cuda --batch-size 8 \
     --threads "$SLURM_CPUS_PER_TASK" --cells "$CELL"
