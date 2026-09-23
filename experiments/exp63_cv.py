"""EXP63 step 2, LOCAL driver -- 12-fold leave-one-month-out test against v6.

Runs the same per-(option, fold) units as the cluster array
(cluster/exp63_task.py, which also documents the hypothesis, the measured facts
and the formulations) in ONE process, so the features are assembled once, then
calls the same collector (cluster/exp63_collect.py). Each finished unit is saved
atomically to cache/exp63/units/; a rerun after a crash skips finished units and
resumes where it stopped.

ONLY RUN THIS IF experiments/exp63_screen.py PASSED ITS GATE (it prints the
options to use). DECISION RULE: adopt an option only if its 12-month paired
bootstrap against v6 has a 95% CI entirely below zero and its effect is clearly
larger than the null control N's.

Prerequisite: cache/exp62_{pb,ptg,spec_P}.npy from experiments/exp62_cv.py
(checked before any fitting).

Measured cost basis (8 threads: 511/1400 fit ~240 s; 6 threads contended ~330 s;
feature build ~1 min, RSS 4.3 GB, 5.5 GB peak with a fit):
    A ~7-9 min/fold, B ~16-20 min/fold, N ~5-6 min/fold at 6 threads
    A+B+N for 12 folds ~5.5-7 h. Prefer the cluster array (~30-45 min wall).

    systemd-inhibit --what=idle:sleep:handle-lid-switch --why=exp63 \\
      systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \\
      /home/thinklex/prc2026-venv/bin/python experiments/exp63_cv.py \\
      --options A:p95:pooled,B:p95:pooled,N 2>&1 | tee -a cache/exp63_cv.log
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prc.config import CACHE
from cluster import exp63_task, exp63_collect


def main():
    ap = argparse.ArgumentParser(description='EXP63 step 2, local')
    ap.add_argument('--options', required=False, default=None,
                    help='comma-separated, as printed by exp63_screen.py, e.g. '
                         'A:p95:pooled,B:p95:pooled,N')
    ap.add_argument('--months', default='1-12')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--spec-row', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        exp63_task.main(['--smoke', '--threads', '4'])
        exp63_collect.main(['--smoke'])
        return
    if not a.options:
        sys.exit('--options is required (use the ones exp63_screen.py printed after PASS)')
    exp63_collect.require_exp62(CACHE)            # fail before hours of fitting
    exp63_task.main(['--options', a.options, '--months', a.months,
                     '--threads', str(a.threads)])
    exp63_collect.main(['--options', a.options, '--spec-row', str(a.spec_row)])


if __name__ == '__main__':
    main()
