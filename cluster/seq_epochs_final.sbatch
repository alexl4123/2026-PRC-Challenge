#!/usr/bin/env bash
#SBATCH --job-name=prc-seqepfin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# seq final fits with EPOCHS epochs (env var, REQUIRED) into cache/seq_ep<EPOCHS>_final/ (--tag; the shipped
# cache/seq_final/ is never touched). Task id = seed, so --array=3-5 adds seeds 3-5.
#
#   sbatch --export=ALL,EPOCHS=20 cluster/seq_epochs_final.sbatch
#   sbatch --export=ALL,EPOCHS=15 --array=3-5 cluster/seq_epochs_final.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# final phase (2026-10-09): sbatch --export=ALL,PRC_RANKING=final ... reads/writes cache_final/
C=cache
if [ "${PRC_RANKING:-}" = final ]; then C=cache_final; fi
mkdir -p logs
[ -n "${EPOCHS:-}" ] || { echo "FATAL: pass --export=ALL,EPOCHS=<n>"; exit 2; }
for f in $C/training.parquet $C/ranking.parquet $C/v11/feat_training.parquet $C/v11/feat_ranking.parquet \
         $C/adsb_lol/features_training.parquet $C/adsb_lol/features_ranking.parquet $C/external/metar \
         $C/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_final.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
SEED=$SLURM_ARRAY_TASK_ID
echo "task $SEED -> seq epochs $EPOCHS final seed $SEED on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_final.py --seed "$SEED" --epochs "$EPOCHS" --tag "seq_ep$EPOCHS" --device cuda \
     --threads "$SLURM_CPUS_PER_TASK"
