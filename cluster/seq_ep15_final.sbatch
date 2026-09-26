#!/usr/bin/env bash
#SBATCH --job-name=prc-seq15fin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# seq with 15 epochs: final fits, seeds 0-2, all of 2025 -> ranking set, into cache/seq_ep15_final/ (--tag; the shipped
# cache/seq_final/ is never touched). Run alongside the folds to save a round trip; used only if the swap eval PASSES.
#
#   sbatch cluster/seq_ep15_final.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/training.parquet cache/ranking.parquet cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet \
         cache/adsb_lol/features_training.parquet cache/adsb_lol/features_ranking.parquet cache/external/metar \
         cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_final.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
SEED=$SLURM_ARRAY_TASK_ID
echo "task $SEED -> seq epochs 15 final seed $SEED on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_final.py --seed "$SEED" --epochs 15 --tag seq_ep15 --device cuda --threads "$SLURM_CPUS_PER_TASK"
