"""EXP37 -- replicate the exp36 result at a second global seed.

The exp36 comparison holds the bulk regressor and the global classifier FIXED
(the same cached arrays feed both arms), so it is already free of most seed
noise, and the winning specialist is either deterministic (table, isotonic) or
averaged over three seeds. What it does NOT establish is whether the SIZE of
the gain depends on which bulk regressor / global classifier it is layered on.
This refits both at a second seed and repeats the comparison end to end.

  /home/thinklex/prc2026-venv/bin/python experiments/exp37_seed_replicate.py 1
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month
from prc.models.tail_harness import (assemble, run_component, mixture,
                                     fp_bulk_regressor, fp_tail_classifier)
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture
pd.set_option('display.width', 220)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MONTHS = [int(a) for a in sys.argv[2:]] or list(range(1, 13))
THREADS = 4

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy()
tail = y > TAIL_CAP
P = population(os_, miss)
sel = np.isin(month, MONTHS)
print(f'seed {SEED}, months {MONTHS}: {sel.sum():,} rows, {int(tail[sel].sum())} tail; '
      f'P has {int(P.sum()):,} rows / {int(tail[P].sum())} tail  ({time.time()-t0:.0f}s)',
      flush=True)

rb = run_component('bulk_reg', X, meta, fp_bulk_regressor, MONTHS, SEED, THREADS)
pT = run_component('clf_T', X, meta, fp_tail_classifier, MONTHS, SEED, THREADS)

print('\n--- specialist, fold-safe ---', flush=True)
p_spec = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    t = time.time()
    s = TailSpecialist(threads=THREADS).fit(X[~va], y[~va], apt[~va], os_[~va], miss[~va])
    pv, Pv = s.predict(X[va], apt[va], os_[va], miss[va])
    p_spec[va] = pv
    print(f'    month {m:02d}: {int(Pv.sum()):,} P-rows  ({time.time()-t:.0f}s)', flush=True)

e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(os_[va]) & (os_[va] > TAIL_CAP), os_[va], tm)

base = mixture(pT, e_tail, rb)
hyb = mixture(apply_to_mixture(pT, p_spec, P), e_tail, rb)

print('\n' + '=' * 100)
print(f'SEED {SEED}, months {MONTHS}')
print('=' * 100)
for nm, p in [('BASE shipped', base), ('HYB  specialist', hyb)]:
    d = decompose(y[sel], p[sel])
    s_ = sel & P
    print(f'  {nm:16s} RMSE {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.2f} | '
          f'tail {d["rmse_tail"]:8.1f} (contrib {d["contrib_tail"]:6.1f}) | '
          f'in-P RMSE {np.sqrt(np.mean((y[s_]-p[s_])**2)):8.1f}')
r = paired_bootstrap(y[sel], hyb[sel], base[sel])
print(f'\n  paired HYB vs BASE: {r["diff"]:+8.2f} +- {r["sd"]:6.2f}  '
      f'CI [{r["lo"]:+8.1f}, {r["hi"]:+8.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
pm_b = per_month(y[sel], base[sel], month[sel])
pm_h = per_month(y[sel], hyb[sel], month[sel])
cmp = pm_b[['month', 'n', 'n_tail', 'rmse']].rename(columns={'rmse': 'BASE'})
cmp['HYB'] = pm_h['rmse'].values
cmp['delta'] = cmp.HYB - cmp.BASE
print('\n' + cmp.round(2).to_string(index=False))
np.save(f'cache/tail/exp37_seed{SEED:02d}.npy',
        np.vstack([base, hyb]).astype(np.float32))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
