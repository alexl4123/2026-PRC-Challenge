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

    # Filenames are r-<regconfig>__c-<clfconfig>__seedNN__mMM.npy (the grid
    # campaign) or <config>__seedNN__mMM.npy (the older campaigns).
    # NOTE: the loop variable is deliberately NOT called `month` -- an earlier
    # version shadowed the month ARRAY built above, so `month == m` later became
    # an int-to-int comparison returning a scalar False, and `p[False]` silently
    # produced a (0, N)-shaped view instead of raising anything sensible.
    parts = defaultdict(dict)
    for f in files:
        bits = f.stem.split('__')
        mm = int(bits[-1][1:])
        seed = bits[-2]
        arm = '__'.join(bits[:-2])
        parts[(arm, seed)][mm] = np.load(f)

    oof = {}
    for (cfg, seed), bym in sorted(parts.items()):
        missing = sorted(set(range(1, 13)) - set(bym))
        if missing:
            print(f'  {cfg}/{seed}: INCOMPLETE, missing months {missing} -- skipped')
            continue
        p = np.full(len(y), np.nan)
        ok = True
        for m, v in bym.items():
            sel = month == m
            if sel.sum() != len(v):
                print(f'  {cfg}/{seed}: month {m} has {sel.sum():,} rows but the task '
                      f'wrote {len(v):,} predictions -- SKIPPED (stale or wrong cache?)')
                ok = False
                break
            p[sel] = v
        if not ok:
            continue
        if not np.isfinite(p).all():
            print(f'  {cfg}/{seed}: {int((~np.isfinite(p)).sum()):,} rows unfilled -- skipped')
            continue
        oof[(cfg, seed)] = p

    print(f'\n{len(oof)} complete OOF sets from {len(files)} task outputs\n')
    rows = []
    for (arm, seed), p in oof.items():
        s = decompose(y, p)
        rows.append(dict(arm=arm, seed=seed, rmse=s['rmse'],
                         bulk=s['rmse_bulk'], tail=s['rmse_tail']))
    df = pd.DataFrame(rows)
    print('--- per (arm, seed) ---')
    print(df.round(2).to_string(index=False))
    print('\n--- per arm, averaged over seeds (single-model RMSE, not an ensemble) ---')
    print(df.groupby('arm')[['rmse', 'bulk', 'tail']].agg(['mean', 'std', 'min'])
          .round(2).to_string())

    # Seed averaging: the 2024 winner gained 3.2% from this alone.
    by_cfg = defaultdict(list)
    for (arm, seed), p in oof.items():
        by_cfg[arm].append(p)
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

    # The comparison the grid exists to make: best arm vs the current baseline.
    if len(by_cfg) > 1:
        print('\n--- paired comparison between arms (seed-averaged) ---')
        avg = {k: np.mean(v, axis=0) for k, v in by_cfg.items()}
        ranked = sorted(avg, key=lambda k: decompose(y, avg[k])['rmse'])
        base = ranked[-1]
        for k in ranked[:-1]:
            r = paired_bootstrap(y, avg[k], avg[base])
            print(f'  {k:24s} vs {base:24s}: {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
                  f'CI [{r["lo"]:+6.1f}, {r["hi"]:+6.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'seeds')
