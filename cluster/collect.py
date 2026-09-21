"""Assemble array-task outputs into OOF predictions and evaluate them.

    python cluster/collect.py seeds
    python cluster/collect.py ablation
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, CACHE
from prc.io import load
from prc.validation import decompose, paired_bootstrap, per_month


def main(campaign):
    d = CACHE / 'cv' / campaign
    files = sorted(d.glob('*.npy'))
    if not files:
        sys.exit(f'no task outputs under {d}')
    tr = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt'])
    y = tr[TARGET].astype(float).to_numpy()
    month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()

    parts = defaultdict(dict)
    for f in files:
        cfg, seed, m = f.stem.split('__')
        parts[(cfg, seed)][int(m[1:])] = np.load(f)

    oof = {}
    for (cfg, seed), bym in sorted(parts.items()):
        missing = sorted(set(range(1, 13)) - set(bym))
        if missing:
            print(f'  {cfg}/{seed}: INCOMPLETE, missing months {missing} -- skipped')
            continue
        p = np.full(len(y), np.nan)
        for m, v in bym.items():
            p[month == m] = v
        oof[(cfg, seed)] = p

    print(f'\n{len(oof)} complete OOF sets from {len(files)} task outputs\n')
    rows = []
    for (cfg, seed), p in oof.items():
        s = decompose(y, p)
        rows.append(dict(config=cfg, seed=seed, rmse=s['rmse'],
                         bulk=s['rmse_bulk'], tail=s['rmse_tail']))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # Seed averaging: the 2024 winner gained 3.2% from this alone.
    by_cfg = defaultdict(list)
    for (cfg, seed), p in oof.items():
        by_cfg[cfg].append(p)
    print()
    for cfg, ps in by_cfg.items():
        if len(ps) < 2:
            continue
        singles = [decompose(y, p)['rmse'] for p in ps]
        print(f'{cfg}: {len(ps)} seeds | single-seed RMSE mean {np.mean(singles):.2f} '
              f'(min {min(singles):.2f}, max {max(singles):.2f})')
        for k in [2, 5, 10, 20]:
            if k <= len(ps):
                avg = np.mean(ps[:k], axis=0)
                print(f'    average of {k:2d} seeds: {decompose(y, avg)["rmse"]:7.2f}')
        best = np.mean(ps, axis=0)
        r = paired_bootstrap(y, best, ps[0])
        print(f'    all-seed average vs seed 0: {r["diff"]:+.2f} +- {r["sd"]:.2f}  '
              f'CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]')
        print(per_month(y, best, month).round(1).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'seeds')
