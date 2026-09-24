#!/usr/bin/env bash
#SBATCH --job-name=prc-w2finnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# W2 final fits, network member: 3 seeds, all of 2025 -> ranking set, v11 columns, emb cap 16
# (cluster/w2_final.py --family nn). --gres=gpu:1 is MANDATORY on -p gpu. --mem 48G: the
# shipped network final used 40G (training and ranking rows are encoded together).
#
#   sbatch cluster/w2_final_nn.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first"; exit 2; }
done
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/w2_final.py --family nn --seed "$SLURM_ARRAY_TASK_ID" --device cuda \
     --threads "$SLURM_CPUS_PER_TASK"
