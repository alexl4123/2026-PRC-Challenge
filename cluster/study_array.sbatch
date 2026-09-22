#!/usr/bin/env bash
#SBATCH --job-name=prc-study
#SBATCH --partition=any
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-215%40
#
# Model-family study: 6 families x 3 seeds x 12 months = 216 tasks.
#
#   sbatch --array=0-215%40 cluster/study_array.sbatch study
#
# Families (see cluster/study_task.py for what each varies):
#   0 mix       shipped v5 mixture -- the BASELINE every other arm is measured against
#   1 routed    four disjoint routes by the NM proxy (the largest identified lever)
#   2 lintree   linear_tree bulk regressor -- leaves that can extrapolate past 7200s
#   3 perapt    ten per-airport bulk regressors
#   4 catboost  CatBoost bulk regressor (ordered target statistics)
#   5 xgboost   XGBoost bulk regressor (ensemble diversity)
#
# 4 cpus x 40 concurrent = 160 of 592 cores (27%). Do NOT raise it: LightGBM
# scales sublinearly past ~4 threads, and the partition is shared.
#
# --time is 1h, not the 25min used elsewhere: catboost/xgboost/perapt are slower
# than a single LightGBM fit and a TIMEOUT wastes the whole task.
set -euo pipefail
CAMPAIGN="${1:-study}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

FAMS=(mix routed lintree perapt catboost xgboost)
i=$SLURM_ARRAY_TASK_ID
MONTH=$(( i % 12 + 1 )); r=$(( i / 12 ))
SEED=$(( r % 3 ));       r=$(( r / 3 ))
FAMILY=${FAMS[$(( r % 6 ))]}

srun python cluster/study_task.py --campaign "$CAMPAIGN" --family "$FAMILY" \
     --seed "$SEED" --month "$MONTH" --threads "$SLURM_CPUS_PER_TASK"
