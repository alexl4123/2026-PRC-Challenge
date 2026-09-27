#!/usr/bin/env bash
#SBATCH --job-name=prc-seqvar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%4
#
# seq fold fits for ONE named variant (env VARIANT, REQUIRED; whitelist below) into cache/cv/seqsw_<VARIANT>/.
# 12 months x 3 seeds, index i -> seed i / 12, month i % 12 + 1. Idempotent; the folder's _config.json pins the settings.
# Overnight grid of 2026-09-27 (judged by experiments/seq/seq_swap_eval.py, rules in its header):
#   sbatch --export=ALL,VARIANT=ep60      cluster/seq_variant_array.sbatch
#   sbatch --export=ALL,VARIANT=ep40_d128 cluster/seq_variant_array.sbatch
#   ... (see the case list)
# --gres=gpu:1 is MANDATORY on -p gpu.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
case "${VARIANT:-}" in
    ep60)      ARGS="--epochs 60" ;;
    ep80)      ARGS="--epochs 80" ;;
    ep40_d128) ARGS="--epochs 40 --d 128" ;;
    ep50_d128) ARGS="--epochs 50 --d 128" ;;
    ep40_l3)   ARGS="--epochs 40 --layers 3" ;;
    ep40_lr3)  ARGS="--epochs 40 --lr 3e-3" ;;
    *) echo "FATAL: VARIANT must be one of ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3 (got '${VARIANT:-}')"; exit 2 ;;
esac
for f in cache/training.parquet cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet \
         cache/external/metar cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_task.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 12 ))
MONTH=$(( i % 12 + 1 ))
echo "task $i -> seq variant $VARIANT ($ARGS) seed $SEED month $MONTH on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_task.py --seed "$SEED" --month "$MONTH" $ARGS --campaign "seqsw_$VARIANT" \
     --device cuda --threads "$SLURM_CPUS_PER_TASK"
