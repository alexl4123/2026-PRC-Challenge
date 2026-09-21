#!/usr/bin/env bash
# Run ON copperhead. Creates a conda env with the stack the experiments need.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh"

if conda env list | grep -q '^prc2026 '; then
    echo "env prc2026 already exists"
else
    conda create -y -n prc2026 python=3.12
fi
conda activate prc2026
pip install -q --upgrade pip
pip install -q "pyarrow>=20" pandas numpy scikit-learn lightgbm
python -c "import pyarrow,pandas,lightgbm,numpy;
print('pyarrow',pyarrow.__version__,'pandas',pandas.__version__,'lgbm',lightgbm.__version__)"
echo "NOTE: the system pyarrow 19 cannot read the challenge parquets"
echo "      ('Repetition level histogram size mismatch'); >=20 is required."
