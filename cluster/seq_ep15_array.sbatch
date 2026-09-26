#!/usr/bin/env bash
#SBATCH --job-name=prc-seq15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%6
#
# seq with 15 epochs (the settings-screen winner, 2026-09-26): 12 months x 3 seeds = 36 fold fits into the SAME folder
# as the screen, cache/cv/seqsw_ep15/ (its _config.json pins epochs 15; the 8 screen folds -- months 3/6/9/12, seeds
# 0-1 -- exist and are skipped, so 28 new fits). SEED-MAJOR: 0-11 seed 0, 12-23 seed 1, 24-35 seed 2. ~5 min each.
# Judged by experiments/seq/seq_swap_eval.py --new seqsw_ep15 (rule in its header).
#
#   sbatch cluster/seq_ep15_array.sbatch
#
# --gres=gpu:1 is MANDATORY on -p gpu. Idempotent.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/training.parquet cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet \
         cache/external/metar cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_task.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 12 ))
MONTH=$(( i % 12 + 1 ))
echo "task $i -> seq epochs 15 seed $SEED month $MONTH on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_task.py --seed "$SEED" --month "$MONTH" --epochs 15 --campaign seqsw_ep15 \
     --device cuda --threads "$SLURM_CPUS_PER_TASK"
