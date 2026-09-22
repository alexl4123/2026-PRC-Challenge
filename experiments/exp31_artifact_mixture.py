"""EXP31 -- retarget the tail classifier at the stamping artifact.

EXP30 established the mechanism precisely:
  * y == off_sched - delta   exactly, where delta = BLOCK_TIME - SCHED_TIME.
  * A = {|delta| <= 60s} has a 9.575% base rate (199,634 rows) against the
    0.028% of the y>7200 target the current classifier is trained on.
  * Conditional on A, y == off_sched to within 60s. So A is a DETERMINISTIC
    route to y, and it has 340x more positives to learn from.
  * The tail is an airport phenomenon: of the 4,944 rows with AOBT_3 missing
    AND off_sched > 7200 (the decision-relevant population), LIRF is 65.9%
    tail, LTFM 7.2%, EGLL 2.1%, and EDDF/EDDM/EHAM/LEBL/LEMD are 0.0% across
    2,319 rows. The "45% coin flip" in the day-one log is a pooling artifact.

So this compares, through identical folds and with the bulk regressor held
fixed where possible:

    BASE   p(y>7200) * e_tail            + (1-p) * E[y|y<=7200]   (the shipped model)
    M1     q(A)      * off_sched         + (1-q) * E[y|not A]
    M1b    q(A)      * off_sched         + (1-q) * E[y|y<=7200]
    M2     q(A)*1{off_sched>7200} * off_sched + ... * E[y|y<=7200]

Run:
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python experiments/exp31_artifact_mixture.py 1 3 7
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month
from prc.models.tail_harness import (assemble, artifact_label, run_component,
                                     fp_bulk_regressor, fp_notA_regressor,
                                     fp_tail_classifier, fp_artifact_classifier,
                                     mixture)
pd.set_option('display.width', 220)

MONTHS = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
SEED = 0
THREADS = 4

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy()
month = meta['month'].to_numpy()
off_sched = meta['off_sched'].to_numpy()
apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy()
A = artifact_label(meta).astype(bool)
sel = np.isin(month, MONTHS)
print(f'{len(y):,} rows; evaluating months {MONTHS} -> {sel.sum():,} rows, '
      f'{int((y[sel] > TAIL_CAP).sum())} tail  ({time.time()-t0:.0f}s)', flush=True)

comp = {}
for name, fn in [('bulk_reg', fp_bulk_regressor), ('clf_T', fp_tail_classifier),
                 ('clf_A', fp_artifact_classifier), ('notA_reg', fp_notA_regressor)]:
    print(f'\n--- component {name} ---', flush=True)
    comp[name] = run_component(name, X, meta, fn, months=MONTHS, seed=SEED, threads=THREADS)

# e_tail exactly as cluster/cv_task.py builds it: per fold, training-set mean.
e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(off_sched[va]) & (off_sched[va] > TAIL_CAP),
                          off_sched[va], tm)

os_safe = np.where(np.isfinite(off_sched), off_sched, 0.0)
pT, qA, rb, rn = comp['clf_T'], comp['clf_A'], comp['bulk_reg'], comp['notA_reg']

preds = {
    'BASE  pT*e_tail+(1-pT)*bulk': mixture(pT, e_tail, rb),
    'M1    qA*os+(1-qA)*notA':     mixture(qA, os_safe, rn),
    'M1b   qA*os+(1-qA)*bulk':     mixture(qA, os_safe, rb),
    'M2    qA*1{os>cap}*os+..bulk': mixture(qA * (os_safe > TAIL_CAP), e_tail, rb),
    'REF   bulk only':             rb,
}

print('\n' + '=' * 110)
print(f'RESULTS on months {MONTHS}')
print('=' * 110)
rows = []
for k, p in preds.items():
    d = decompose(y[sel], p[sel])
    rows.append(dict(model=k, rmse=d['rmse'], bulk=d['rmse_bulk'], tail=d['rmse_tail'],
                     c_bulk=d['contrib_bulk'], c_tail=d['contrib_tail'], n_tail=d['n_tail']))
res = pd.DataFrame(rows).sort_values('rmse')
print(res.round(2).to_string(index=False))

base = preds['BASE  pT*e_tail+(1-pT)*bulk']
print('\n--- paired bootstrap vs BASE (negative = better) ---')
for k, p in preds.items():
    if k.startswith('BASE'):
        continue
    r = paired_bootstrap(y[sel], p[sel], base[sel])
    print(f'  {k:30s} {r["diff"]:+9.2f} +- {r["sd"]:6.2f}  CI [{r["lo"]:+8.1f}, '
          f'{r["hi"]:+8.1f}]  P(worse)={r["p_a_worse"]*100:5.1f}%')

# ---------------------------------------------------------------------------
print('\n' + '=' * 110)
print('CLASSIFIER DIAGNOSTICS on the decision-relevant population')
print('   (AOBT_3 missing AND off_sched > 7200 -- 4,944 rows full year, 450 tail)')
print('=' * 110)
look = sel & miss & (os_safe > TAIL_CAP)
print(f'  n={int(look.sum()):,}  tail={int((y[look] > TAIL_CAP).sum())} '
      f'({(y[look] > TAIL_CAP).mean()*100:.1f}%)   A={int(A[look].sum())} '
      f'({A[look].mean()*100:.1f}%)')
out = []
for a in sorted(set(apt[look])):
    s = look & (apt == a)
    out.append(dict(apt=a, n=int(s.sum()), p_tail=(y[s] > TAIL_CAP).mean() * 100,
                    pA=A[s].mean() * 100, mean_pT=pT[s].mean() * 100,
                    mean_qA=qA[s].mean() * 100,
                    pT_on_tail=pT[s & (y > TAIL_CAP)].mean() * 100 if (s & (y > TAIL_CAP)).any() else np.nan,
                    qA_on_tail=qA[s & (y > TAIL_CAP)].mean() * 100 if (s & (y > TAIL_CAP)).any() else np.nan,
                    pT_on_bulk=pT[s & (y <= TAIL_CAP)].mean() * 100 if (s & (y <= TAIL_CAP)).any() else np.nan,
                    qA_on_bulk=qA[s & (y <= TAIL_CAP)].mean() * 100 if (s & (y <= TAIL_CAP)).any() else np.nan))
print(pd.DataFrame(out).round(2).to_string(index=False))

print('\n--- calibration: mean predicted vs actual, by predicted decile (all eval rows) ---')
for nm, p, truth in [('pT vs y>7200', pT, (y > TAIL_CAP)), ('qA vs A', qA, A)]:
    idx = np.where(sel)[0]
    q = pd.qcut(pd.Series(p[idx]).rank(method='first'), 10, labels=False)
    tab = pd.DataFrame({'dec': q, 'p': p[idx], 't': truth[idx].astype(float)})
    g = tab.groupby('dec').agg(n=('p', 'size'), mean_p=('p', 'mean'), rate=('t', 'mean'))
    print(f'  {nm}:  overall mean_p={p[idx].mean():.6f} actual={truth[idx].mean():.6f} '
          f'ratio={p[idx].mean()/max(truth[idx].mean(),1e-12):.2f}')
    print('    top decile: mean_p={:.5f} rate={:.5f}  n={}'.format(
        g.mean_p.iloc[-1], g.rate.iloc[-1], int(g.n.iloc[-1])))

print('\n--- precision/recall at the top of each score ---')
for nm, p, truth in [('pT -> y>7200', pT, (y > TAIL_CAP)), ('qA -> y>7200', qA, (y > TAIL_CAP))]:
    ps, ts = p[sel], truth[sel]
    print(f'  {nm}')
    for k in [50, 150, 300, 600, 1500]:
        if k > len(ps):
            continue
        thr = np.partition(ps, -k)[-k]
        s = ps >= thr
        print(f'    top {k:5d}: precision {ts[s].mean()*100:6.2f}%  recall '
              f'{ts[s].sum()}/{int(ts.sum())}')

print('\n--- where the squared error lives (BASE vs best alternative) ---')
best = res.iloc[0]['model']
for k in ['BASE  pT*e_tail+(1-pT)*bulk', best]:
    p = preds[k]
    e2 = (y - p) ** 2
    tot = e2[sel].sum()
    print(f'  {k}: total SSE {tot:.3e}')
    for nm, s in [('true tail (y>7200)', sel & (y > TAIL_CAP)),
                  ('bulk, LIRF+miss+os>cap', sel & (y <= TAIL_CAP) & miss & (os_safe > TAIL_CAP) & (apt == 'LIRF')),
                  ('bulk, other miss+os>cap', sel & (y <= TAIL_CAP) & miss & (os_safe > TAIL_CAP) & (apt != 'LIRF')),
                  ('bulk, rest', sel & (y <= TAIL_CAP) & ~(miss & (os_safe > TAIL_CAP)))]:
        print(f'      {nm:26s} n={int(s.sum()):9,}  SSE {e2[s].sum():.3e} '
              f'({e2[s].sum()/tot*100:5.1f}%)  rmse {np.sqrt(e2[s].mean()):9.1f}')

np.save('cache/tail/exp31_preds.npy',
        np.vstack([preds[k] for k in preds]).astype(np.float32))
with open('cache/tail/exp31_keys.txt', 'w') as f:
    f.write('\n'.join(preds))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
