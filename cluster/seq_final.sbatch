#!/usr/bin/env bash
#SBATCH --job-name=prc-seqfin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# Bet D: final fits of the 'seq' member, seeds 0-2, all of 2025 -> the ranking set (cluster/seq_final.py),
# written into cache/seq_final/seq__seedNN.npy (bulk prediction, ranking DEP order) + ranking_ids.npy.
# Only worth running once experiments/seq/seq_eval.py PASSES on the fold fits (cluster/seq_array.sbatch).
#
#   sbatch cluster/seq_final.sbatch
#   sbatch --export=ALL,SEQ_FAMILY=seqnc cluster/seq_final.sbatch     # no-context twin (only if the info arm is used)
#   sbatch --export=ALL,SEQ_C1=1 cluster/seq_final.sbatch             # C1 feature set -> cache/seq_c1_final/
#
# --gres=gpu:1 is MANDATORY on -p gpu. Idempotent: a finished seed is skipped. Laptop smoke peaked at 6.7 GB.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
FAMILY="${SEQ_FAMILY:-seq}"
case "$FAMILY" in seq) FLAG="" ;; seqnc) FLAG="--no-context" ;; *) echo "FATAL: SEQ_FAMILY must be seq or seqnc"; exit 2 ;; esac
NEED="cache/training.parquet cache/ranking.parquet cache/v11/feat_training.parquet cache/v11/feat_ranking.parquet"
NEED="$NEED cache/adsb_lol/features_training.parquet cache/adsb_lol/features_ranking.parquet cache/external/metar"
NEED="$NEED cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_final.py"
if [ "${SEQ_C1:-0}" = "1" ]; then
    FLAG="$FLAG --deice"
    NEED="$NEED cache/deice_0926/ind_training.parquet cache/deice_0926/ind_ranking.parquet"
fi
for f in $NEED; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done

SEED=$SLURM_ARRAY_TASK_ID
echo "task $SEED -> $FAMILY final seed $SEED (flags: ${FLAG:-none}) on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

srun python cluster/seq_final.py --seed "$SEED" $FLAG --device cuda --threads "$SLURM_CPUS_PER_TASK"
