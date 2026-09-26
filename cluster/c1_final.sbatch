#!/usr/bin/env bash
#SBATCH --job-name=prc-c1fin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-14%5
#
# C1 CONSOLIDATED RETRAIN (2026-09-26), final fits, CPU members: 5 families x 3 seeds, all of 2025 -> ranking set,
# A3 arm-2 columns + de-icing shares (cluster/w2_final.py ... --tag c1) -> cache/c1_final/. Run alongside the CV
# to save a round trip (minutes each); only used if experiments/adsb/a3_eval.py --old adsb2 --new c1 PASSES.
#
#   sbatch cluster/c1_final.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet cache/adsb_lol/features_training.parquet \
         cache/adsb_lol/features_ranking.parquet cache/deice_0926/ind_training.parquet cache/deice_0926/ind_ranking.parquet; do
    [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first"; exit 2; }
done
FAMS=(mix perapt lintree xgboost routed)
i=$SLURM_ARRAY_TASK_ID
FAM=${FAMS[$(( i / 3 ))]}
SEED=$(( i % 3 ))
echo "task $i -> family $FAM seed $SEED"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026          # the CPU env: has xgboost (prc2026gpu does not)
srun python cluster/w2_final.py --family "$FAM" --seed "$SEED" --threads "$SLURM_CPUS_PER_TASK" \
     --adsb --extra --deice --nn-missing sentinel --tag c1
