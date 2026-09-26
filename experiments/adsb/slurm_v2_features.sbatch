#!/usr/bin/env bash
#SBATCH --job-name=prc-adsbv2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#
# ADS-B v2 features (label-free extra columns for flights whose off-block is not seen), full scale:
#   1. adsb_events_v2.py for every day with points + events (8 in parallel; measured on the laptop: 9-16 s per day,
#      peak 1.0 GB per worker) -> cache/adsb_lol/{events_v2,calib_v2,grid_v2}/<date>.parquet
#   2. back-calculation curves from the 2025 calibration samples -> cache/adsb_lol/v2_calib_curves.parquet
#   3. features_v2_training.parquet (2,085,047 rows) and features_v2_ranking.parquet, keyed by MVT_ID_mvt.
# Reads the production outputs (points/, events/, stands.parquet) and never modifies them.
# Estimated runtime (estimate, not measured at full scale): 429 days x ~13 s / 8 workers ~ 12 min + matching and
# feature assembly ~10-20 min.
#
#   sbatch experiments/adsb/slurm_v2_features.sbatch
#
# Idempotent: days whose events_v2 file exists are skipped (pass --redo to rebuild them).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/training.parquet cache/ranking.parquet cache/adsb_lol/stands.parquet; do
  [ -f "$f" ] || { echo "FATAL: $f missing"; exit 2; }
done
n_ev=$(ls cache/adsb_lol/events/*.parquet 2>/dev/null | wc -l)
echo "events files present: $n_ev (expected 429)"
[ "$n_ev" -gt 0 ] || { echo "FATAL: no cache/adsb_lol/events/*.parquet -- run slurm_stage2_features.sbatch first"; exit 2; }
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026
/usr/bin/time -v srun python experiments/adsb/adsb_build_features_v2.py --workers "$SLURM_CPUS_PER_TASK" "$@" 2>&1 \
    | grep -v -E '^\s+(Page size|Exit status|Swaps|File system|Socket|Signals|Average|Voluntary|Involuntary|Minor|Major)'
ls -la cache/adsb_lol/features_v2_training.parquet cache/adsb_lol/features_v2_ranking.parquet
