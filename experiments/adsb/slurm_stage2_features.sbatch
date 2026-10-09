#!/usr/bin/env bash
#SBATCH --job-name=prc-adsb2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # the gpu partition rejects jobs without a GPU request; the "any" nodes are drained
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.out
#
# adsb.lol stage 2 + features, after the stage-1 array has finished:
#   events/<date>.parquet for every day with points (8 in parallel, ~6-10 s per day measured on the laptop)
#   features_training.parquet (2,085,047 departures) and features_ranking.parquet, keyed by MVT_ID_mvt.
# Needs cache/training.parquet and cache/ranking.parquet (already on the cluster, see cluster/README.md).
# Estimated runtime: 429 days x ~8 s / 8 workers ~ 7 min + matching ~5 min (estimate, not measured at full scale).
#
#   sbatch --dependency=afterany:<stage1 array job id> experiments/adsb/slurm_stage2_features.sbatch
#
# It reports missing days (their departures get adsb_day_ok = 0) instead of failing, so it can be
# run on a partial archive; re-run after filling the gaps.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# final phase (2026-10-09): sbatch --export=ALL,PRC_RANKING=final ... reads/writes cache_final/
C=cache
if [ "${PRC_RANKING:-}" = final ]; then C=cache_final; fi
mkdir -p logs
for f in $C/training.parquet $C/ranking.parquet; do
  [ -f "$f" ] || { echo "FATAL: $f missing -- rsync it first (cluster/sync.sh)"; exit 2; }
done
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026
python experiments/adsb/adsb_days.py --missing | head -50 | sed 's/^/missing points: /'
srun python experiments/adsb/adsb_build_features.py --workers "$SLURM_CPUS_PER_TASK" "$@"
