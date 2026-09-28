#!/usr/bin/env bash
#SBATCH --job-name=prc-a2finnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-5
#
# A3 ARM 2 (ADS-B + wake + backlog + gap cap, cluster/w2_extra.py): final fits, network members: 0-2 = nn seeds 0-2, 3-5 = w3 seeds 0-2, all of 2025 -> ranking set,
# W2 + adsb.lol columns (cluster/w2_final.py --adsb) -> cache/adsb2_final/. --gres=gpu:1 is MANDATORY.
#
#   sbatch --dependency=afterok:<stage-2 job> cluster/adsb2_final_nn.sbatch
# More seeds: sbatch --export=ALL,SEED_BASE=3 cluster/adsb2_final_nn.sbatch   (seeds 3-5; default 0 = seeds 0-2)
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet cache/adsb_lol/features_training.parquet cache/adsb_lol/features_ranking.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing"; exit 2; }
done
FAMS=(nn w3)
i=$SLURM_ARRAY_TASK_ID
FAM=${FAMS[$(( i / 3 ))]}
SEED=$(( ${SEED_BASE:-0} + i % 3 ))      # SEED_BASE=3 -> seeds 3-5 (2026-09-28 member-seed batch)
echo "task $i -> $FAM seed $SEED"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/w2_final.py --family "$FAM" --seed "$SEED" --device cuda --threads "$SLURM_CPUS_PER_TASK" --adsb --extra
