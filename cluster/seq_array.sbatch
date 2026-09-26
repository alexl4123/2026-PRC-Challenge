#!/usr/bin/env bash
#SBATCH --job-name=prc-seq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-35%4
#
# Bet D: the 'seq' member (set model over neighbouring movements, prc/models/seq_net.py), 12 months x 3 seeds
# = 36 fold fits (cluster/seq_task.py) with the adsb2 members' 143 columns, written into
# cache/cv/seq/seq__seedNN__mMM.npy (+ .bulk.npy). Index order is SEED-MAJOR: 0-11 = seed 0, 12-23 seed 1,
# 24-35 seed 2 -- so a partial run still gives complete seeds.
#
#   sbatch cluster/seq_array.sbatch                                          # seq, all 36
#   sbatch --array=0-11%4 cluster/seq_array.sbatch                           # seq, seed 0 only (early read)
#   sbatch --export=ALL,SEQ_FAMILY=seqnc --array=0-23%4 cluster/seq_array.sbatch   # no-context twin, seeds 0-1
#   sbatch --export=ALL,SEQ_C1=1 cluster/seq_array.sbatch                    # C1 feature set -> cache/cv/seq_c1/
#
# SEQ_FAMILY=seqnc runs the ablation twin: the same network without the context branch (a plain MLP, minutes per
# task); seq - seqnc isolates what the context adds (experiments/seq/seq_eval.py, info arm).
# --gres=gpu:1 is MANDATORY on -p gpu (devices are confined per cgroup); seq_task.py stops at once if no GPU is
# visible. --mem 80G: laptop smoke peaked at 5.7 GB (frame ~4.5 GB + movement table + design); generous on a
# 2 TB node. Idempotent: finished folds are skipped; a campaign folder refuses tasks with other settings.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
FAMILY="${SEQ_FAMILY:-seq}"
case "$FAMILY" in seq) FLAG="" ;; seqnc) FLAG="--no-context" ;; *) echo "FATAL: SEQ_FAMILY must be seq or seqnc"; exit 2 ;; esac
NEED="cache/training.parquet cache/v11/feat_training.parquet cache/adsb_lol/features_training.parquet"
NEED="$NEED cache/external/metar cache/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_task.py"
if [ "${SEQ_C1:-0}" = "1" ]; then
    FLAG="$FLAG --deice --campaign seq_c1"
    NEED="$NEED cache/deice_0926/ind_training.parquet"
fi
for f in $NEED; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done

i=$SLURM_ARRAY_TASK_ID
SEED=$(( i / 12 ))
MONTH=$(( i % 12 + 1 ))
echo "task $i -> $FAMILY seed $SEED month $MONTH (flags: ${FLAG:-none}) on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu

srun python cluster/seq_task.py --seed "$SEED" --month "$MONTH" $FLAG --device cuda --threads "$SLURM_CPUS_PER_TASK"
