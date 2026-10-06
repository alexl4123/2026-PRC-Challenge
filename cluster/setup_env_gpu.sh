#!/usr/bin/env bash
# Run ON the cluster. Creates a SEPARATE env for the GPU node, because the CPU
# env's torch (if any) would be the CPU build and silently ignore the cards.
#
# The L40S is compute capability sm_89 (Ada), which needs a CUDA 12.x build.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"

if conda env list | grep -q '^prc2026gpu '; then
    echo "env prc2026gpu already exists"
else
    conda create -y -n prc2026gpu python=3.12
fi
conda activate prc2026gpu
pip install -q --upgrade pip
pip install -q "pyarrow>=20" pandas numpy scikit-learn lightgbm
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print('torch', torch.__version__, '| cuda build', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('devices:', torch.cuda.device_count(),
          '|', torch.cuda.get_device_name(0),
          '| capability', torch.cuda.get_device_capability(0))
else:
    print('NOTE: on the login node this is expected -- devices are confined per')
    print('      cgroup, so the login node itself sees none. Verify on a GPU node.')
    print('      Use the ABSOLUTE interpreter path: this script activated the')
    print('      env in its own subshell, so your shell is still on base and a')
    print('      bare `python` under srun would be the wrong one.')
    print()
    print('  srun -p gpu --gres=gpu:1 -t 00:05:00 \\')
    print('    ~/miniconda3/envs/prc2026gpu/bin/python \\')
    print('    -c "import torch;print(torch.cuda.get_device_name(0))"')
PY
