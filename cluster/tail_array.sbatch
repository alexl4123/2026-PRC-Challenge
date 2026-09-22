#!/usr/bin/env bash
#SBATCH --job-name=prc-tailspec
#SBATCH --partition=any
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-239%40
#
# Confirm the tail specialist on the cluster, on the grid's best arm.
#
#   sbatch --array=0-239%40 cluster/tail_array.sbatch tailspec
#
# 2 arms x 10 seeds x 12 months = 240 tasks, all on r-full / c-base (the arm the
# 480-task grid picked: 287.81 seed-averaged, and the most stable one).
#   arm 0 -> spec-r-full__c-base__seedNN__mMM.npy   (with the tail specialist)
#   arm 1 ->      r-full__c-base__seedNN__mMM.npy   (matched baseline)
# Both land in the same campaign directory, so `python cluster/collect.py
# tailspec` scores them as two arms and paired-bootstraps one against the other.
#
# 4 cpus x 40 concurrent = 160 of 592 cores (27%). Do not raise this: LightGBM
# scales sublinearly past ~4 threads, so a wider array costs fairshare and buys
# no wall time. See claude_log 2026-09-21 section 16.
#
# Locally measured expectation (12 folds, base features, seed 0, uncapped OOF):
#   baseline 294.39 -> specialist 281.32, paired -12.39 +- 6.12, CI [-25.0, -2.9]
# If the cluster's r-full arm does NOT show a negative difference with a CI
# clear of zero, the local result did not transfer -- say so and stop, do not
# ship it.
set -euo pipefail
CAMPAIGN="${1:-tailspec}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

i=$SLURM_ARRAY_TASK_ID
MONTH=$(( i % 12 + 1 )); r=$(( i / 12 ))
SEED=$(( r % 10 )); r=$(( r / 10 ))
ARM=$(( r % 2 ))
CONFIG=full
CLF=base

EXTRA=()
if [ "$ARM" -eq 1 ]; then EXTRA+=(--no-specialist); fi

srun python cluster/tail_task.py --campaign "$CAMPAIGN" --config "$CONFIG" \
     --clf-config "$CLF" --seed "$SEED" --month "$MONTH" \
     --threads "$SLURM_CPUS_PER_TASK" "${EXTRA[@]}"
