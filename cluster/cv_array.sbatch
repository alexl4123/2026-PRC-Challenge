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
#   sbatch --array=0-239%40 cluster/cv_array.sbatch seeds      # 20 seeds x 12 months
#   sbatch --array=0-47%40   cluster/cv_array.sbatch ablation   # 4 configs x 12 months
#   sbatch --array=0-479%60  cluster/cv_array.sbatch grid       # the overnight sweep
set -euo pipefail
CAMPAIGN="${1:-seeds}"

# Work from wherever sbatch was invoked rather than a hardcoded path, so the
# clone can live anywhere (it is under ~/<repo>
# on the cluster login node, not ~/prc2026).
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
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
      CONFIG=${CONFIGS[$(( i / 12 ))]}; SEED=0; MONTH=$(( i % 12 + 1 )); CLF=$CONFIG ;;
  grid)
      # The overnight campaign. Crosses three open questions at once:
      #   regressor feature set   {cong, full}          2
      #   classifier feature set  {base, cong}          2
      #   seed                    0..9                 10
      #   month                   1..12                12
      # = 480 tasks. Answers, in one sweep: does a rich regressor with a lean
      # classifier beat matched feature sets (EXP15/EXP16); how much does seed
      # averaging buy; and which combination is best per month.
      REGS=(cong full); CLFS=(base cong)
      MONTH=$(( i % 12 + 1 )); r=$(( i / 12 ))
      SEED=$(( r % 10 )); r=$(( r / 10 ))
      CLF=${CLFS[$(( r % 2 ))]}; CONFIG=${REGS[$(( r / 2 ))]} ;;
  *) echo "unknown campaign: $CAMPAIGN" >&2; exit 1 ;;
esac

srun python cluster/cv_task.py --campaign "$CAMPAIGN" --config "$CONFIG" \
     --clf-config "$CLF" --seed "$SEED" --month "$MONTH" \
     --threads "$SLURM_CPUS_PER_TASK"
