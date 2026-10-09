#!/usr/bin/env bash
#SBATCH --job-name=prc-w2fin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-14%5
#
# W2 final fits, CPU members: 5 families x 3 seeds, all of 2025 -> ranking set
# (cluster/w2_final.py). Only worth running once experiments/w2/w2_eval.py has PASSED, or
# alongside the CV to save a round trip (cheap: minutes each).
#
#   sbatch cluster/w2_final.sbatch
#
# Writes cache/w2_final/<family>__seedNN.npy + ranking_ids.npy. Idempotent. Needs
# cache/v11/feat_{training,ranking}.parquet on the cluster.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# final phase (2026-10-09): sbatch --export=ALL,PRC_RANKING=final ... reads/writes cache_final/
C=cache
if [ "${PRC_RANKING:-}" = final ]; then C=cache_final; fi
mkdir -p logs
for f in $C/v11/feat_training.parquet $C/v11/feat_ranking.parquet; do
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
srun python cluster/w2_final.py --family "$FAM" --seed "$SEED" --threads "$SLURM_CPUS_PER_TASK"
