#!/usr/bin/env bash
#SBATCH --job-name=prc-exp63
#SBATCH --partition=any
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
#
# EXP63 step 2 -- the three-way target treatment, one task per (option, fold).
# RUN ONLY IF experiments/exp63_screen.py PASSED ITS GATE; it prints the exact
# options to pass. The options are the positional arguments, in order:
#
#   sbatch --array=0-35%40 cluster/exp63_array.sbatch A:p95:pooled B:p95:pooled N
#
# task i -> option index i/12 (in the order given), month i%12+1.
# The array size MUST be 12 x (number of options) - 1: 1 option -> 0-11,
# 2 -> 0-23, 3 -> 0-35. There is deliberately no default --array.
#
#   A:<bnd>:<region>  three-component soft mixture       ~10-20 min / task
#   B:<bnd>:<region>  v6 regressor + nested OOF q feature ~25-45 min / task
#                     (12 classifier fits: 11 inner + 1 outer)
#   N                 null control: v6 regressor, seed+1000 ~8-15 min / task
#
# Each task writes cache/exp63/units/<tag>__s00__mMM.npz and SKIPS if it exists,
# so a partially failed array can be resubmitted as is (or with only the indices
# cluster/exp63_collect.py names as missing). Peak memory measured locally:
# ~5.5 GB for the feature build plus one 511-leaf fit; 10G leaves headroom.
#
# 4 cpus x up to 36 concurrent = 144 of 592 cores. %40 throttle as elsewhere;
# LightGBM scales sublinearly past ~4 threads, so do not widen the tasks.
#
# The tasks need NO exp62 cache: the collector attaches the v6 tail branch.
# Collect LOCALLY (where cache/exp62_*.npy live) after pulling the units back.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as an array: sbatch --array=0-N%40 cluster/exp63_array.sbatch OPTIONS...}"
if [ "$#" -lt 1 ]; then
    echo "FATAL: no options given. Example: sbatch --array=0-35%40 $0 A:p95:pooled B:p95:pooled N" >&2
    exit 2
fi
OPTS=("$@")

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

i=$SLURM_ARRAY_TASK_ID
MONTH=$(( i % 12 + 1 ))
K=$(( i / 12 ))
if [ "$K" -ge "${#OPTS[@]}" ]; then
    echo "FATAL: array index $i maps to option #$K but only ${#OPTS[@]} option(s) were given;" \
         "use --array=0-$(( ${#OPTS[@]} * 12 - 1 ))%40" >&2
    exit 2
fi
OPTION=${OPTS[$K]}
echo "task $i -> option $OPTION, month $MONTH, host $(hostname), $(date -Is)"

srun python cluster/exp63_task.py --options "$OPTION" --months "$MONTH" \
     --threads "$SLURM_CPUS_PER_TASK"
