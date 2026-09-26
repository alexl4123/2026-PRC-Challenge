#!/usr/bin/env bash
#SBATCH --job-name=prc-c1finnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-5
#
# C1 CONSOLIDATED RETRAIN (2026-09-26), final fits, network members: 0-2 = nn seeds 0-2 (sentinel NaN), 3-5 = w3
# seeds 0-2, all of 2025 -> ranking set -> cache/c1_final/. --gres=gpu:1 is MANDATORY.
#
#   sbatch cluster/c1_final_nn.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet cache/adsb_lol/features_training.parquet \
         cache/adsb_lol/features_ranking.parquet cache/deice_0926/ind_training.parquet cache/deice_0926/ind_ranking.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first"; exit 2; }
done
FAMS=(nn w3)
i=$SLURM_ARRAY_TASK_ID
FAM=${FAMS[$(( i / 3 ))]}
SEED=$(( i % 3 ))
echo "task $i -> $FAM seed $SEED"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/w2_final.py --family "$FAM" --seed "$SEED" --device cuda --threads "$SLURM_CPUS_PER_TASK" \
     --adsb --extra --deice --nn-missing sentinel --tag c1
