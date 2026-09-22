#!/usr/bin/env bash
# Run ON the cluster, from the repository root, BEFORE submitting any array.
# Checks every prerequisite an array task needs. Exits non-zero on the first
# real problem, so a missing file costs five seconds instead of 480 failed tasks.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== conda environment =="
if source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
   || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null; then
    ok "conda found"
else
    bad "no conda profile script; cannot activate an environment"
fi
if conda activate prc2026 2>/dev/null; then
    ok "env prc2026 activated"
    PYV=$(python -c 'import pyarrow;print(pyarrow.__version__)' 2>/dev/null || echo none)
    case "$PYV" in
        none) bad "pyarrow not importable" ;;
        1[0-9].*|[0-9].*) bad "pyarrow $PYV is too old; >=20 required (older versions fail on
             the challenge parquets with 'Repetition level histogram size mismatch')" ;;
        *) ok "pyarrow $PYV" ;;
    esac
    for m in pandas numpy lightgbm; do
        v=$(python -c "import $m;print($m.__version__)" 2>/dev/null) \
            && ok "$m $v" || bad "$m not importable"
    done
    # Only the study campaign needs these; warn rather than fail so the other
    # campaigns still pass preflight on an older environment.
    for m in catboost xgboost; do
        v=$(python -c "import $m;print($m.__version__)" 2>/dev/null) \
            && ok "$m $v" \
            || printf '  \033[33mwarn\033[0m %s\n' \
               "$m missing -- needed only by cluster/study_array.sbatch; rerun cluster/setup_env.sh"
    done
else
    bad "conda env 'prc2026' missing -- run: bash cluster/setup_env.sh"
fi

echo
echo "== data (git does NOT carry this; rsync it from the laptop) =="
for f in cache/training.parquet cache/ranking.parquet cache/external/atfm_daily.csv; do
    if [[ -s "$f" ]]; then ok "$f ($(du -h "$f" | cut -f1))"
    else bad "$f MISSING"; fi
done
n=$(ls cache/external/metar/*.parquet 2>/dev/null | wc -l)
if [[ "$n" -eq 10 ]]; then ok "cache/external/metar/ has all 10 stations"
else bad "cache/external/metar/ has $n/10 station files -- weather features will try to
       DOWNLOAD at runtime, which fails on a compute node without internet"; fi

echo
echo "== row counts (a truncated cache silently corrupts every result) =="
if [[ -s cache/training.parquet ]]; then
    python - <<'PY' || fail=1
import pyarrow.parquet as pq, sys
want = {'cache/training.parquet': 4_167_797, 'cache/ranking.parquet': 689_534}
bad = False
for f, w in want.items():
    try:
        n = pq.ParquetFile(f).metadata.num_rows
    except Exception as e:
        print(f'  \033[31mFAIL\033[0m {f}: {e}'); bad = True; continue
    if n == w:
        print(f'  \033[32mok\033[0m   {f}: {n:,} rows')
    else:
        print(f'  \033[31mFAIL\033[0m {f}: {n:,} rows, expected {w:,}'); bad = True
sys.exit(1 if bad else 0)
PY
fi

echo
echo "== slurm =="
command -v sbatch >/dev/null && ok "sbatch present" || bad "sbatch not on PATH"
[[ -f cluster/cv_array.sbatch ]] && ok "cluster/cv_array.sbatch" || bad "sbatch script missing"
mkdir -p logs && ok "logs/ ready"

echo
if [[ "$fail" -eq 0 ]]; then
    echo -e "\033[32mALL CHECKS PASSED\033[0m -- safe to submit:"
    echo "  sbatch --array=0-479%60 cluster/cv_array.sbatch grid"
else
    echo -e "\033[31mPREFLIGHT FAILED\033[0m -- fix the above before submitting."
fi
exit "$fail"
