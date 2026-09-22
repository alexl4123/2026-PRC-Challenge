"""EXP47 -- WHERE does the network differ from the trees?

A blend that helps is only actionable if you can say what it is fixing. This
slices the bulk-row residuals of the best tree (lgb900), the network, and the
blend by the cuts that matter operationally: airport, whether the NM pushback
record AOBT_3 exists, and how far the flight is from a normal off_aobt3.

Reads cached fold predictions only -- no fitting, seconds to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TAIL_CAP
from prc.models.altfam import CACHE, SCREEN_MONTHS, design, load_folds, screen
from prc.validation import rmse

y = np.load(CACHE / 'altfam' / 'y_screen.npy')
month = np.load(CACHE / 'altfam' / 'month_screen.npy')
X, _, mon_all, _ = design()
sel = screen(mon_all)
Xs = X.loc[sel, ['APT', 'aobt_missing', 'off_aobt3']].reset_index(drop=True)
del X

P = {}
for lab, nm in [('lgb350', 'lgb_bulk'), ('lgb900', 'lgb900_bulk_s0'),
                ('cat', 'catboost_bulk_s0'), ('xgb', 'xgboost_bulk_s0'),
                ('nn', 'nn_bulk_s0')]:
    v = load_folds(nm, month)
    if v is not None:
        P[lab] = v
if 'lgb900' in P and 'nn' in P:
    P['blend'] = 0.47 * P['lgb900'] + 0.53 * P['nn']

b = y <= TAIL_CAP
yb = y[b]
have = list(P)
print(f'screening rows {len(y):,}, bulk {int(b.sum()):,}, models {have}\n')


def table(name, groups):
    rows = []
    for g, m in groups:
        if m.sum() < 200:
            continue
        rows.append({name: g, 'n': int(m.sum()),
                     **{k: rmse(yb[m], P[k][b][m]) for k in have}})
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    for k in have:
        if k not in ('lgb900', 'nn', 'blend'):
            continue
    if 'lgb900' in df and 'nn' in df:
        df['nn-lgb900'] = df['nn'] - df['lgb900']
    if 'lgb900' in df and 'blend' in df:
        df['blend-lgb900'] = df['blend'] - df['lgb900']
    print(df.round(2).to_string(index=False))
    print()


apt = Xs.APT.astype(str).to_numpy()[b]
table('airport', [(a, apt == a) for a in sorted(set(apt))])

am = Xs.aobt_missing.to_numpy()[b]
table('aobt_missing', [('present', am == 0), ('missing', am == 1)])

oa = Xs.off_aobt3.to_numpy(dtype=np.float64, na_value=np.nan)[b]
q = np.array([-np.inf, 600, 900, 1200, 1800, 3600, np.inf])
labels = ['<600', '600-900', '900-1200', '1200-1800', '1800-3600', '>3600']
table('off_aobt3', [(labels[i], np.isfinite(oa) & (oa > q[i]) & (oa <= q[i + 1]))
                    for i in range(len(labels))] + [('AOBT_3 absent', ~np.isfinite(oa))])

mb = month[b]
table('month', [(str(m), mb == m) for m in SCREEN_MONTHS])
