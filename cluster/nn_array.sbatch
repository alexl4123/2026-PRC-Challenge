#!/usr/bin/env bash
#SBATCH --job-name=prc-nn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%8
#
# Entity-embedding network, 3 seeds x 12 months = 36 tasks.
#
#   sbatch --array=0-35%8 cluster/nn_array.sbatch study
#
# WHY THIS CAN RUN IMMEDIATELY: coppernode-gpu01 is OUTSIDE the `any*`
# partition, so this never contends with the CPU campaigns or with anyone
# else's `broadwell`/`sunnycove` work. The CPU study can be queued and this
# started at the same time.
#
# --gres=gpu:1 IS MANDATORY. The cluster confines devices per cgroup, so a job
# in -p gpu WITHOUT it sees no devices at all (`nvidia-smi` returns "No devices
# were found", exit 6). That is the confinement, not a broken node. The GRES is
# untyped, so `--gres=gpu:N` -- `--gres=gpu:l40s:N` does not resolve.
# cluster/nn_task.py refuses to start on CPU when a device was expected, so this
# mistake fails in seconds instead of training slowly for two hours.
#
# %8 caps concurrency at the node's 8 cards, one task per card.
set -euo pipefail
CAMPAIGN="${1:-study}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

i=$SLURM_ARRAY_TASK_ID
MONTH=$(( i % 12 + 1 ))
SEED=$(( i / 12 ))

srun python cluster/nn_task.py --campaign "$CAMPAIGN" --seed "$SEED" \
     --month "$MONTH" --device cuda --threads "$SLURM_CPUS_PER_TASK"
