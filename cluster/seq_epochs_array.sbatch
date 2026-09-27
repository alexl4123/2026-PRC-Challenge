#!/usr/bin/env bash
#SBATCH --job-name=prc-seqep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%4
#
# seq fold fits with EPOCHS epochs (env var, REQUIRED) into cache/cv/seqsw_ep<EPOCHS>/ (the folder's _config.json pins
# the settings). Index i -> seed i / 12, month i % 12 + 1, so --array=0-35 = seeds 0-2, --array=36-71 = seeds 3-5.
# Idempotent: existing folds are skipped (the 15-epoch screen / campaign folds are reused).
#
#   sbatch --export=ALL,EPOCHS=20 cluster/seq_epochs_array.sbatch                    # 20 epochs, seeds 0-2
#   sbatch --export=ALL,EPOCHS=15 --array=36-71%4 cluster/seq_epochs_array.sbatch    # 15 epochs, seeds 3-5
#
# --gres=gpu:1 is MANDATORY on -p gpu.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
[ -n "${EPOCHS:-}" ] || { echo "FATAL: pass --export=ALL,EPOCHS=<n>"; exit 2; }
for f in cache/training.parquet cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet \
         cache/external/metar cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_task.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 12 ))
MONTH=$(( i % 12 + 1 ))
echo "task $i -> seq epochs $EPOCHS seed $SEED month $MONTH on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_task.py --seed "$SEED" --month "$MONTH" --epochs "$EPOCHS" --campaign "seqsw_ep$EPOCHS" \
     --device cuda --threads "$SLURM_CPUS_PER_TASK"
