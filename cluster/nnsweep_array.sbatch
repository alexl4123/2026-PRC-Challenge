#!/usr/bin/env bash
#SBATCH --job-name=prc-nnsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-3
#
# The default --array matches the DEFAULT ARGUMENTS (campaign nnsweep, stage
# fixed, group A), so a bare `sbatch cluster/nnsweep_array.sbatch` does exactly
# the right first step: prep the four screening folds. Every other submission
# passes --array explicitly; `python cluster/nnsweep_configs.py` prints the
# range for each group. An index past the end of a group exits immediately with
# the correct range rather than silently wrapping.
#
# ============================================================================
# ENTITY-EMBEDDING NETWORK -- HYPERPARAMETER SWEEP
#
#   sbatch --array=0-3    cluster/nnsweep_array.sbatch nnsweep fixed A
#   sbatch --array=0-87%8 cluster/nnsweep_array.sbatch nnsweep sweep A
#
# Arguments: <campaign> <stage> <group>.  Defaults: nnsweep fixed A.
#
# ALWAYS RUN `fixed` FIRST AND LET IT FINISH. The sweep stage refuses to start
# without it and prints this command, so the failure costs seconds rather than
# a whole array. Get the exact --array ranges from
#
#   python cluster/nnsweep_configs.py
#
# which prints them per group and verifies that the index mapping is a
# bijection. The mapping lives in that file and NOT in this one: doing
# `MONTH=$(( i % 12 + 1 ))` in bash, as cluster/nn_array.sbatch does, puts the
# mapping in two places that can drift apart. Here the task resolves its own
# --index.
# ============================================================================
#
# --gres=gpu:1 IS MANDATORY. The cluster confines devices per cgroup, so a job
# in -p gpu WITHOUT it sees no devices at all (`nvidia-smi` returns "No devices
# were found", exit 6). That is the confinement, not a broken node. The GRES is
# untyped, so `--gres=gpu:l40s:1` does not resolve -- always plain `gpu:N`.
# nnsweep_task.py resolves the device BEFORE it touches any data and raises if
# CUDA was asked for and is unavailable, so this mistake fails in seconds
# instead of training on eight CPU cores for four hours.
#
# %8 caps concurrency at the node's 8 L40S cards, one task per card. Do not
# raise it. coppernode-gpu01 has 64 cores, so 8 tasks x 8 cpus is exactly the
# node; more concurrency would oversubscribe the CPUs that feed the cards.
#
# WHY THIS CAN RUN IMMEDIATELY: coppernode-gpu01 is OUTSIDE the `any*`
# partition, so it never contends with the CPU campaigns.
#
# --time=04:00:00 covers both stages. A `fixed` task does a full assemble(), a
# 300-round classifier and the production 1,400-round / 511-leaf LightGBM
# reference; a `sweep` task only trains a network, because the prep stage
# cached everything else. The longest sweep arms are `ep40`/`ep40lr6` (3.3x the
# epochs) and `b2k` (4x the optimiser steps). Every task prints its own fit and
# total seconds -- read the first completed log and resize if these are wrong.
#
# I/O. Each sweep task reads its fold's ~530 MB numeric block, so one wave of 8
# pulls ~4 GB off the home filesystem. With 2 TB of host RAM the node's page
# cache holds all four folds after the first wave, so only that wave pays for
# it. This is also why the sweep stage is cheap enough to run 88 tasks: the
# expensive per-fold work happened once, in the prep stage.
#
# DISK. The prep stage caches ~1.2 GB per fold per numeric-design mode under
# cache/cv/<campaign>/_fixed/ (group A needs both 'median' and 'sentinel', so
# ~4.7 GB for four folds). It is rebuildable; delete the directory when the
# campaign is scored.
set -euo pipefail
CAMPAIGN="${1:-nnsweep}"
STAGE="${2:-fixed}"
GROUP="${3:-A}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

case "$STAGE" in
    fixed)
        # CPU-only work. It holds a card it does not use for ~1 h; with four
        # tasks on an eight-card node that is acceptable and keeps the whole
        # campaign in one submission. To spare the cards instead:
        #   sbatch -p any --gres=none --cpus-per-task=8 --mem=32G \
        #          --array=0-3 cluster/nnsweep_array.sbatch nnsweep fixed A
        srun python cluster/nnsweep_task.py --campaign "$CAMPAIGN" --stage fixed \
             --group "$GROUP" --index "$SLURM_ARRAY_TASK_ID" \
             --threads "$SLURM_CPUS_PER_TASK"
        ;;
    sweep)
        srun python cluster/nnsweep_task.py --campaign "$CAMPAIGN" --stage sweep \
             --group "$GROUP" --index "$SLURM_ARRAY_TASK_ID" --device cuda \
             --threads "$SLURM_CPUS_PER_TASK"
        ;;
    *)
        echo "unknown stage '$STAGE' -- expected 'fixed' or 'sweep'" >&2
        exit 2
        ;;
esac
