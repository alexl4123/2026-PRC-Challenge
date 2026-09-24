#!/usr/bin/env bash
#SBATCH --job-name=prc-f1ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=1-12%4
#
# F1: fine-tune TabICL as Rome's tail-zone classifier, one held-out month per task
# (experiments/tfm/f1_finetune_rome.py). Each task runs all four variants x 3 seeds.
#
#   sbatch cluster/f1_array.sbatch
#
# -p gpu, not `any`: `any` has MaxTime=01:00:00 and a longer request sits PENDING
# with (PartitionTimeLimit). --gres=gpu:1 is MANDATORY on -p gpu (per-cgroup devices).
#
# ONE-TIME SETUP on the cluster (see the commands in claude_log/2026-09-24_dev.log §17):
#   - tabicl 2.2.0 + transformers 5.17.0 in the prc2026gpu env
#   - the TabICL checkpoint under ~/.cache/huggingface/hub/models--jingang--TabICL
#     (compute nodes are run offline: HF_HUB_OFFLINE=1 below)
#   - inputs: cache/major/{dep_training.parquet,v9s_oof.npy},
#             cache/{exp62_cand_training.parquet,exp62_spec_P.npy,exp62_pb.npy}
# Idempotent: finished (variant, month, seed) parts in cache/tfm_f1/parts/ are skipped,
# so a failed task can simply be resubmitted with --array=<its index>.
# Pull the parts back and build the report on the laptop:
#   rsync -az vivaldi:03_2026_PRC_Challenge/2026-PRC-Challenge/cache/tfm_f1/parts/ \
#         <laptop repo>/cache/tfm_f1/parts/
#   python experiments/tfm/f1_finetune_rome.py --report --seeds 0 1 2
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

for f in cache/major/dep_training.parquet cache/major/v9s_oof.npy cache/exp62_cand_training.parquet \
         cache/exp62_spec_P.npy cache/exp62_pb.npy; do
    [ -f "$f" ] || { echo "FATAL: missing input $f -- rsync it from the laptop first"; exit 2; }
done
ls "$HOME"/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/tabicl-classifier-v2-20260212.ckpt >/dev/null \
    || { echo "FATAL: TabICL checkpoint not in ~/.cache/huggingface/hub -- rsync it first"; exit 2; }

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026gpu
export HF_HUB_OFFLINE=1

srun python experiments/tfm/f1_finetune_rome.py --device cuda --batch-size 8 \
     --threads "$SLURM_CPUS_PER_TASK" --months "$SLURM_ARRAY_TASK_ID" --seeds 0 1 2
