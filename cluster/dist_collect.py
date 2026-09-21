"""Assemble dist_task.py outputs into OOF predictions and score them.

Differs from collect.py in two ways:

* each task file holds SEVERAL prediction vectors (one per integration variant),
  so a config expands into config x variant rows;
* every row is compared against the cached v3-mixture OOF from EXP13 with a
  PAIRED bootstrap, because an absolute OOF RMSE has a bootstrap sd of ~47 s and
  cannot resolve anything the leaderboard cares about. A config is only reported
  as better if its 95% interval clears zero.

    python cluster/dist_collect.py dist
    python cluster/dist_collect.py seeds --seed-average
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, CACHE
from prc.io import load
from prc.validation import decompose, paired_bootstrap, per_month

pd.set_option('display.width', 210)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign', nargs='?', default='dist')
    ap.add_argument('--seed-average', action='store_true',
                    help='average across seeds within each (config, variant)')
    ap.add_argument('--top', type=int, default=8, help='how many rows to detail')
    a = ap.parse_args()

    d = CACHE / 'cv' / a.campaign
    files = sorted(d.glob('*.npz'))
    if not files:
        sys.exit(f'no task outputs under {d}')

    tr = load('training', phase='DEP', columns=[TARGET, 'MVT_TIME_UTC_mvt'])
    y = tr[TARGET].astype(float).to_numpy()
    month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()

    base = None
    if (CACHE / 'exp13_oof.npy').exists():
        base = np.load(CACHE / 'exp13_oof.npy')[2]   # row 2 = v3 mixture, 296.22
        b = decompose(y, base)
        print(f'baseline (EXP13 v3 mixture): OOF {b["rmse"]:.2f} | bulk '
              f'{b["rmse_bulk"]:.1f} | tail {b["rmse_tail"]:.0f}\n')

    # (config, seed, variant) -> {month: preds}
    parts = defaultdict(dict)
    for f in files:
        cfg, seed, m = f.stem.split('__')
        z = np.load(f, allow_pickle=False)
        for v, p in zip([str(s) for s in z['variants']], z['pred']):
            parts[(cfg, seed, v)][int(m[1:])] = p

    oof = {}
    for key, bym in sorted(parts.items()):
        missing = sorted(set(range(1, 13)) - set(bym))
        if missing:
            print(f'  {key}: INCOMPLETE, missing months {missing} -- skipped')
            continue
        p = np.full(len(y), np.nan)
        for m, v in bym.items():
            p[month == m] = v
        oof[key] = p

    if a.seed_average:
        bysv = defaultdict(list)
        for (cfg, seed, v), p in oof.items():
            bysv[(cfg, 'avg', v)].append(p)
        oof = {k: np.mean(ps, axis=0) for k, ps in bysv.items() if len(ps) > 1} or oof

    rows = []
    for (cfg, seed, v), p in oof.items():
        s = decompose(y, p)
        r = dict(config=cfg, seed=seed, variant=v, rmse=s['rmse'],
                 bulk=s['rmse_bulk'], tail=s['rmse_tail'])
        if base is not None:
            pb = paired_bootstrap(y, p, base)
            r.update(vs_base=pb['diff'], sd=pb['sd'], lo=pb['lo'], hi=pb['hi'],
                     verdict='BETTER' if pb['hi'] < 0 else
                             'worse' if pb['lo'] > 0 else 'ns')
        rows.append(r)

    res = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    print(f'{len(oof)} complete OOF sets from {len(files)} task files\n')
    print(res.round(2).to_string(index=False))

    if base is not None and 'verdict' in res:
        won = res[res.verdict == 'BETTER']
        print(f'\n{len(won)} of {len(res)} configurations beat the baseline with a '
              f'95% paired interval clear of zero.')
    best = res.iloc[0]
    print(f'\nper-month for the best row ({best.config} / {best.variant}):')
    print(per_month(y, oof[(best.config, best.seed, best.variant)], month)
          .round(1).to_string(index=False))


if __name__ == '__main__':
    main()
