"""EXP23 -- how much RMSE does discretising the target cost, before any fitting?

A binned model cannot resolve detail finer than its bins, so predicting every
row at the centroid of its own TRUE bin is a hard floor on what the
architecture in prc/models/distributional.py can reach. It depends only on the
labels and the grid, it costs one pass over y, and it should be checked before
spending 45 minutes on a softmax rather than after.

The result decided the grid in DEFAULT_GRID, and it is not the obvious one:
quantile bins -- the reflexive choice for a skewed target -- are the worst
option at any class count, because they put a few seconds of width at the mode
where resolution buys nothing and thousands of seconds at the 99.9th percentile
where it decides the metric. Since LightGBM grows one tree per class per
iteration, class count is the compute budget, and a fixed linear grid over the
core with a geometric continuation reaches a lower floor at half the classes.

Runtime ~30 s.
  /home/thinklex/prc2026-venv/bin/python experiments/exp23_bin_grid_floor.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.io import load, peak_rss_gb
from prc.models.distributional import make_edges, bin_codes, centroids, Y_MAX
pd.set_option('display.width', 210)

y = load('training', phase='DEP', columns=[TARGET])[TARGET].astype(float).to_numpy()
tail = y > TAIL_CAP
print(f'{len(y):,} departures | {int(tail.sum())} above {TAIL_CAP}s | max {y.max():.0f}')


def floor(edges):
    """RMSE of an oracle that knows each row's bin but nothing finer."""
    c = centroids(y, edges)          # in-sample on purpose: this is a FLOOR
    p = c[bin_codes(y, edges)]
    return dict(K=len(edges) - 1,
                floor=float(np.sqrt(np.mean((y - p) ** 2))),
                bulk=float(np.sqrt(np.mean((y[~tail] - p[~tail]) ** 2))),
                tail=float(np.sqrt(np.mean((y[tail] - p[tail]) ** 2))))


def quantile_edges(n_bulk, n_tail):
    b = np.quantile(y[y <= TAIL_CAP], np.linspace(0, 1, n_bulk + 1))
    b[0], b[-1] = -1.0, TAIL_CAP
    return np.unique(np.concatenate([b, np.geomspace(TAIL_CAP, Y_MAX, n_tail + 1)[1:]]))


rows = []
for nb in (24, 48, 96, 192, 384):
    for nt in (4, 10):
        rows.append(dict(grid=f'quantile nb={nb} nt={nt}', **floor(quantile_edges(nb, nt))))
for cw, ch in ((200, 4000), (150, 4200), (120, 3600), (100, 4000), (75, 3000), (50, 3000)):
    for nt in (1, 3, 10):
        rows.append(dict(grid=f'linear w={cw} hi={ch} nt={nt}',
                         **floor(make_edges(core_width=float(cw), core_hi=float(ch),
                                            n_mid=6, n_tail=nt))))
df = pd.DataFrame(rows).sort_values('K')
print('\n' + df.round(2).to_string(index=False))

best_q = df[df.grid.str.startswith('quantile')].sort_values('bulk').iloc[0]
best_l = df[df.grid.str.startswith('linear')].sort_values('bulk').iloc[0]
print(f'\nbest quantile bulk floor {best_q.bulk:.1f} at K={int(best_q.K)}; '
      f'best linear bulk floor {best_l.bulk:.1f} at K={int(best_l.K)}')
print('A floor of f seconds costs sqrt(rmse^2 + f^2) - rmse against a model at '
      '243 s OOF:')
for f in (28.9, 43.2, 66.4, 87.2):
    print(f'   floor {f:5.1f} s -> +{np.hypot(243.2, f) - 243.2:5.2f} s')
print(f'\npeak RSS {peak_rss_gb():.2f} GB')
