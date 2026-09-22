"""EXP39 -- does seed averaging already buy what the tail specialist buys?

This is the question that decides whether the specialist is worth shipping.

EXP36 measured -12.4 +- 6.1s OOF at seed 0, and a follow-up forensic showed that
83% of the gain comes from FOUR rows: LIRF, AOBT_3 missing, off_sched 58,563 /
73,260 / 59,640 / 64,438 s, all genuine tail rows, where the global classifier
returned p = 0.024 / 0.201 / 0.402 / 0.459 and an isotonic fit on the same
population returns 1.0 (the observed rate for LIRF x AOBT-missing x off_sched >
40,000 is 36 of 36 over the full year).

A one-sided catastrophic miss on four rows is EXACTLY the failure mode that seed
averaging repairs -- the grid found averaging buys 1-4s and that it buys most
where the classifier is noisiest. If two seeds already rescue those rows, the
specialist is redundant. If they do not, it is not.

Uses the cached components at seeds 0 and 1; fits nothing but the specialist.

  /home/thinklex/prc2026-venv/bin/python experiments/exp39_seed_average.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month
from prc.models.tail_harness import assemble, TAILDIR, mixture
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture
pd.set_option('display.width', 220)

SEEDS = [0, 1]
MONTHS = list(range(1, 13))
THREADS = 4

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy()
tail = y > TAIL_CAP
P = population(os_, miss)

missing = [f'{c}__seed{s:02d}__m{m:02d}.npy' for s in SEEDS for m in MONTHS
           for c in ('bulk_reg', 'clf_T')
           if not (TAILDIR / f'{c}__seed{s:02d}__m{m:02d}.npy').exists()]
if missing:
    sys.exit('FATAL: missing cached components, run exp36 (seed 0) and '
             f'exp37_seed_replicate.py 1 first:\n  ' + '\n  '.join(missing[:8]))

comp = {}
for s in SEEDS:
    rb = np.full(len(y), np.nan); pt = np.full(len(y), np.nan)
    for m in MONTHS:
        va = month == m
        rb[va] = np.load(TAILDIR / f'bulk_reg__seed{s:02d}__m{m:02d}.npy')
        pt[va] = np.load(TAILDIR / f'clf_T__seed{s:02d}__m{m:02d}.npy')
    comp[s] = (rb, pt)
print(f'loaded components for seeds {SEEDS}  ({time.time()-t0:.0f}s)', flush=True)

p_spec = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    sp = TailSpecialist(threads=THREADS).fit(X[~va], y[~va], apt[~va], os_[~va], miss[~va])
    p_spec[va], _ = sp.predict(X[va], apt[va], os_[va], miss[va])
print(f'specialist fitted through 12 folds  ({time.time()-t0:.0f}s)', flush=True)

e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(os_[va]) & (os_[va] > TAIL_CAP), os_[va], tm)

preds = {}
for s in SEEDS:
    rb, pt = comp[s]
    preds[f'BASE seed{s}'] = mixture(pt, e_tail, rb)
    preds[f'HYB  seed{s}'] = mixture(apply_to_mixture(pt, p_spec, P), e_tail, rb)
preds['BASE avg2'] = np.mean([preds[f'BASE seed{s}'] for s in SEEDS], axis=0)
preds['HYB  avg2'] = np.mean([preds[f'HYB  seed{s}'] for s in SEEDS], axis=0)
# averaging the PROBABILITIES rather than the predictions -- the grid's framing
pt_avg = np.mean([comp[s][1] for s in SEEDS], axis=0)
rb_avg = np.mean([comp[s][0] for s in SEEDS], axis=0)
preds['BASE avg-p'] = mixture(pt_avg, e_tail, rb_avg)
preds['HYB  avg-p'] = mixture(apply_to_mixture(pt_avg, p_spec, P), e_tail, rb_avg)

print('\n' + '=' * 100)
print('RESULTS (12 folds, 2,085,047 rows, 584 tail)')
print('=' * 100)
rows = []
for k, p in preds.items():
    d = decompose(y, p); s_ = P
    rows.append(dict(model=k, rmse=d['rmse'], bulk=d['rmse_bulk'], tail=d['rmse_tail'],
                     inP=float(np.sqrt(np.mean((y[s_] - p[s_]) ** 2)))))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print('\n--- the comparison that matters: specialist ON TOP OF seed averaging ---')
for a, b in [('HYB  seed0', 'BASE seed0'), ('HYB  seed1', 'BASE seed1'),
             ('HYB  avg2', 'BASE avg2'), ('HYB  avg-p', 'BASE avg-p'),
             ('BASE avg2', 'BASE seed0'), ('BASE avg-p', 'BASE seed0'),
             ('HYB  avg2', 'BASE seed0')]:
    r = paired_bootstrap(y, preds[a], preds[b])
    print(f'  {a:11s} vs {b:11s} {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
          f'CI [{r["lo"]:+7.1f}, {r["hi"]:+7.1f}]  P(worse)={r["p_a_worse"]*100:5.1f}%')

print('\n--- the four rows the specialist rescues at seed 0: what does seed 1 say? ---')
e0 = (y - preds['BASE seed0']) ** 2
eh = (y - preds['HYB  seed0']) ** 2
mv = np.where(P & (np.abs(eh - e0) > 1e9))[0]
t = pd.DataFrame({'apt': apt[mv], 'month': month[mv], 'y': y[mv], 'off_sched': os_[mv],
                  'pT_s0': comp[0][1][mv], 'pT_s1': comp[1][1][mv],
                  'pT_avg': pt_avg[mv], 'p_spec': p_spec[mv], 'is_tail': tail[mv]})
print(t.round(3).to_string(index=False))

print('\n--- per-month, HYB avg2 vs BASE avg2 ---')
pb = per_month(y, preds['BASE avg2'], month)
ph = per_month(y, preds['HYB  avg2'], month)
c = pb[['month', 'n', 'n_tail', 'rmse']].rename(columns={'rmse': 'BASE'})
c['HYB'] = ph['rmse'].values; c['delta'] = c.HYB - c.BASE
print(c.round(2).to_string(index=False))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
