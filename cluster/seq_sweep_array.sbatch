#!/usr/bin/env bash
#SBATCH --job-name=prc-seqsw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-47%6
#
# seq hyper-parameter SCREEN (2026-09-26): 6 configurations x 4 screening months (3, 6, 9, 12) x 2 seeds = 48 fold
# fits, each into its own campaign folder cache/cv/seqsw_<name>/ (the _config.json guard keeps them apart).
# The baseline (K 64, window 5400 s, d 64, 2 layers, 10 epochs) already exists in cache/cv/seq (seeds 0-2).
# SEED-MAJOR: 0-23 = seed 0 (config-major, 4 months each), 24-47 = seed 1.
# Judged by experiments/seq/seq_sweep_eval.py (rule in its header). One fold = ~4 min at K 64; K 128 ~2x.
#
#   sbatch cluster/seq_sweep_array.sbatch
#
# --gres=gpu:1 is MANDATORY on -p gpu. Idempotent: finished folds are skipped.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
for f in cache/training.parquet cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet \
         cache/external/metar cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_task.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
NAMES=(k128 w7200 k128w7200 d128 l3 ep15)
ARGS=("--K 128" "--window 7200" "--K 128 --window 7200" "--d 128" "--layers 3" "--epochs 15")
MONTHS=(3 6 9 12)
i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 24 ))
REM=$(( i % 24 ))
C=$(( REM / 4 ))
MONTH=${MONTHS[$(( REM % 4 ))]}
NAME=${NAMES[$C]}
echo "task $i -> config $NAME (${ARGS[$C]}) seed $SEED month $MONTH on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_task.py --seed "$SEED" --month "$MONTH" ${ARGS[$C]} --campaign "seqsw_$NAME" \
     --device cuda --threads "$SLURM_CPUS_PER_TASK"
