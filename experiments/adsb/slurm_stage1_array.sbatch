#!/usr/bin/env bash
#SBATCH --job-name=prc-adsb1
#SBATCH --partition=gpu   # CPU-only on the gpu nodes (as adsb2_final.sbatch); the "any" nodes are drained (2026-10)
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-428%8
#
# adsb.lol stage 1: ONE DAY per task. Streams that day's globe_history tar straight from GitHub
# (no local tar on disk), keeps points within 12 km / 6000 ft of the ten airports and writes
#     cache/adsb_lol/points/<date>.parquet     (~10-17 MB per day; ~7 GB for all 429 days)
# Task i processes line i+1 of `python experiments/adsb/adsb_days.py` (429 days:
# 2024-12-31..2025-12-31, 2026-01-01..01-31, 2026-06-30..07-31).
#
# Measured on the laptop (July 2025 days, 3.5 GB each): download 149-176 s at 17-24 MB/s, parse 50-70 s on
# 6 workers (~365 CPU-s), main-process peak RSS 0.86 GB. With --stream, download and parse overlap, so 4 CPUs
# (~90 s of parsing) keep pace with the download; more CPUs would idle on the network.
# 50 min / 6 GB is generous; `any` caps jobs at 1 h.
# %8 keeps at most 8 simultaneous downloads from GitHub (be polite; ~1.3 TB in total).
#
# PREREQUISITE (check once, the compute nodes need outbound HTTPS to github.com):
#     srun -p any -t 00:02:00 curl -sI https://github.com | head -1        # expect: HTTP/2 200
#
# Idempotent: a day whose points file exists is skipped, so after failures simply resubmit
# the same array (or only the missing indices, see adsb_days.py --missing).
#
#   sbatch experiments/adsb/slurm_stage1_array.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs cache/adsb_lol/points cache/adsb_lol/meta

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

i=$SLURM_ARRAY_TASK_ID
DATE=$(python experiments/adsb/adsb_days.py | sed -n "$(( i + 1 ))p")
[ -n "$DATE" ] || { echo "FATAL: no date for index $i"; exit 2; }
echo "task $i -> $DATE on $(hostname) at $(date -u +%FT%TZ)"
if [ -f "cache/adsb_lol/points/$DATE.parquet" ]; then echo "$DATE already done"; exit 0; fi

# the release list (meta/PREFERRED_RELEASES_<year>.txt) is refreshed at most daily, written atomically
t0=$(date +%s)
srun python experiments/adsb/adsb_extract_day.py "$DATE" --stream --workers "$SLURM_CPUS_PER_TASK" --skip-events
echo "$DATE done in $(( $(date +%s) - t0 )) s"
