#!/usr/bin/env bash
#SBATCH --job-name=prc-dist
#SBATCH --partition=any
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-71%30
#
# One array task = one (config, seed, month) triple for the EXP20/EXP21
# architectures. The index arithmetic lives in dist_task.py rather than here,
# because the campaigns are no longer a clean rectangle; ask it for the bound:
#
#   python cluster/dist_task.py --campaign dist  --size   # -> --array=0-71
#   python cluster/dist_task.py --campaign arch  --size   # -> --array=0-35
#   python cluster/dist_task.py --campaign seeds --size   # -> --array=0-119
#
# and then match --array to what it prints:
#
#   sbatch --array=0-71%30  cluster/dist_array.sbatch dist
#   sbatch --array=0-35%30  cluster/dist_array.sbatch arch
#   sbatch --array=0-119%30 cluster/dist_array.sbatch seeds
#
# Resources differ from cv_array.sbatch on purpose. A K=47 softmax holds a
# num_data x num_class score matrix plus gradients and hessians -- roughly
# 1.9M x 47 x 16 bytes = 1.4 GB on top of the feature matrix -- so 12G is not
# enough, and one fit is ~12-15 min against ~55 s for the L2 regressor. The
# non-distributional campaigns (`arch`) finish in ~2 min and are simply
# over-provisioned by these settings, which is cheaper than a second file.
set -euo pipefail
CAMPAIGN="${1:-dist}"
mkdir -p logs

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate prc2026

cd "$HOME/prc2026"
srun python cluster/dist_task.py --campaign "$CAMPAIGN" \
     --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
