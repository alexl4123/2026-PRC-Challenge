#!/usr/bin/env bash
#SBATCH --job-name=prc-cv
#SBATCH --partition=any
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-239%40
#
# One array task = one (config, seed, month) triple.
# Timeout-bounded at 25 min per task, per the admin guidance against long jobs.
#
#   sbatch cluster/cv_array.sbatch seeds     # 20 seeds x 12 months, config=full
#   sbatch --array=0-47 cluster/cv_array.sbatch ablation
set -euo pipefail
CAMPAIGN="${1:-seeds}"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

i=$SLURM_ARRAY_TASK_ID
case "$CAMPAIGN" in
  seeds)
      # 20 seeds x 12 months, all on the full feature set
      SEED=$(( i / 12 )); MONTH=$(( i % 12 + 1 )); CONFIG=full ;;
  ablation)
      # 4 configs x 12 months, seed 0
      CONFIGS=(base cong wx full)
      CONFIG=${CONFIGS[$(( i / 12 ))]}; SEED=0; MONTH=$(( i % 12 + 1 )) ;;
  *) echo "unknown campaign: $CAMPAIGN" >&2; exit 1 ;;
esac

cd "$HOME/prc2026"
srun python cluster/cv_task.py --campaign "$CAMPAIGN" --config "$CONFIG" \
     --seed "$SEED" --month "$MONTH" --threads "$SLURM_CPUS_PER_TASK"
