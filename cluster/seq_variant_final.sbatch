#!/usr/bin/env bash
#SBATCH --job-name=prc-seqvarfin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# seq final fits (seeds 0-2, all of 2025 -> ranking set) for ONE named variant (env VARIANT, REQUIRED; same whitelist
# as seq_variant_array.sbatch) into cache/seq_<VARIANT>_final/ (--tag; the shipped folders are never touched).
#   sbatch --export=ALL,VARIANT=ep60 cluster/seq_variant_final.sbatch
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# final phase (2026-10-09): sbatch --export=ALL,PRC_RANKING=final ... reads/writes cache_final/
C=cache
if [ "${PRC_RANKING:-}" = final ]; then C=cache_final; fi
mkdir -p logs
case "${VARIANT:-}" in
    ep60)      ARGS="--epochs 60" ;;
    ep80)      ARGS="--epochs 80" ;;
    ep40_d128) ARGS="--epochs 40 --d 128" ;;
    ep50_d128) ARGS="--epochs 50 --d 128" ;;
    ep40_l3)   ARGS="--epochs 40 --layers 3" ;;
    ep40_lr3)  ARGS="--epochs 40 --lr 3e-3" ;;
    *) echo "FATAL: VARIANT must be one of ep60 ep80 ep40_d128 ep50_d128 ep40_l3 ep40_lr3 (got '${VARIANT:-}')"; exit 2 ;;
esac
for f in $C/training.parquet $C/ranking.parquet $C/v11/feat_training.parquet $C/v11/feat_ranking.parquet \
         $C/adsb_lol/features_training.parquet $C/adsb_lol/features_ranking.parquet $C/external/metar \
         $C/external/atfm_daily.csv prc/models/seq_net.py cluster/seq_final.py; do
    [ -e "$f" ] || { echo "FATAL: $f missing -- git pull / rsync it first"; exit 2; }
done
SEED=$SLURM_ARRAY_TASK_ID
echo "task $SEED -> seq variant $VARIANT ($ARGS) final seed $SEED on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
srun python cluster/seq_final.py --seed "$SEED" $ARGS --tag "seq_$VARIANT" --device cuda --threads "$SLURM_CPUS_PER_TASK"
