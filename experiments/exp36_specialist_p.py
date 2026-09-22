"""EXP36 -- a specialist classifier for the one cell that decides the tail.

EXP32/35 localised the problem exactly. Define the decision-relevant population

    P = {AOBT_3 missing} and {off_sched > 7200}      4,944 rows, 450 tail (9.1%)

P holds 77% of all tail rows AND essentially all of the false-positive damage.
Outside P the tail rate is 0.0065% and the mixture's tail term is inert.

Inside P the base rate is 9.1% rather than 0.028%, so a classifier fitted on P
alone sees 450 positives in 4,944 rows instead of 435 in 1.9M. EXP09's
AOBT-missing SPECIALIST MODEL failed because it replaced the global model and
so could not touch the tail rows outside its population. This does not replace
anything: the shipped mixture runs everywhere, and only P(T|x) inside P is
swapped. Rows outside P keep the shipped prediction exactly.

Ceilings measured in exp35 for the LIRF sub-cell (627 rows): constant p 8,893,
perfect p per off_sched bucket 6,182, oracle label 4,299.

  /home/thinklex/prc2026-venv/bin/python experiments/exp36_specialist_p.py 1 3 7
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month
from prc.models.tail_harness import (assemble, TAILDIR, mixture, run_component,
                                     fp_bulk_regressor, fp_tail_classifier, params, CLF)
from prc.features import CATS
pd.set_option('display.width', 230)

MONTHS = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
SEED, THREADS = 0, 4

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy()
tail = y > TAIL_CAP
P = miss & np.isfinite(os_) & (os_ > TAIL_CAP)
sel = np.isin(month, MONTHS)
print(f'P: n={int(P.sum()):,} tail={int(tail[P].sum())} ({tail[P].mean()*100:.1f}%)  '
      f'| outside P: n={int((~P).sum()):,} tail={int(tail[~P].sum())} '
      f'({tail[~P].mean()*100:.4f}%)   ({time.time()-t0:.0f}s)', flush=True)

print('\n--- baseline components ---', flush=True)
rb = run_component('bulk_reg', X, meta, fp_bulk_regressor, MONTHS, SEED, THREADS)
pT = run_component('clf_T', X, meta, fp_tail_classifier, MONTHS, SEED, THREADS)

# Features available inside P: everything derived from AOBT_3/EOBT/LOBT is NaN
# there, so the specialist gets only what actually varies.
SPEC = ['APT', 'PIER', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt',
        'off_sched', 'log_off_sched', 'hour', 'dow']
SPEC_CATS = [c for c in CATS if c in SPEC]
SPEC_PAR = dict(CLF, learning_rate=0.05, num_leaves=15, min_data_in_leaf=30,
                feature_fraction=0.9, bagging_fraction=0.9)


def fit_spec(tr_mask, va_mask, mono=False, rounds=200, seed=SEED):
    p = params(SPEC_PAR, seed, THREADS)
    if mono:
        p = dict(p, monotone_constraints=[1 if c in ('off_sched', 'log_off_sched') else 0
                                          for c in SPEC])
    k = tr_mask & P
    m = lgb.train(p, lgb.Dataset(X.loc[k, SPEC], tail[k].astype(int),
                                 categorical_feature=SPEC_CATS), num_boost_round=rounds)
    return m.predict(X.loc[va_mask & P, SPEC])


def iso_est(tr_mask, va_mask):
    """Isotonic regression of the tail rate on off_sched, fitted separately for
    LIRF and everything else. The only structure exp35 found inside the cell
    that is both strong and monotone is 'later scheduled departure -> more
    likely a stamping artifact', and isotonic imposes exactly that with no
    other assumption."""
    from sklearn.isotonic import IsotonicRegression
    out = np.full(int((va_mask & P).sum()), np.nan)
    grp_va = np.where(apt[va_mask & P] == 'LIRF', 'LIRF', 'OTHER')
    for g in ['LIRF', 'OTHER']:
        k = tr_mask & P & ((apt == 'LIRF') if g == 'LIRF' else (apt != 'LIRF'))
        if k.sum() < 30:
            out[grp_va == g] = tail[tr_mask & P].mean()
            continue
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(os_[k], tail[k].astype(float))
        m = grp_va == g
        if m.any():
            out[m] = ir.predict(os_[va_mask & P][m])
    return out


def table_est(tr_mask, va_mask, prior_w=8.0):
    """Empirical P(tail | APT, off_sched bucket) with Beta shrinkage to the
    airport rate and then to the global rate. 4,944 rows will not support more
    structure than this, and it is the estimator the GBM has to beat."""
    edges = np.array([7200, 8000, 9000, 10000, 12000, 15000, 20000, 30000, np.inf])
    b = np.digitize(os_, edges) - 1
    ktr = tr_mask & P
    g = tail[ktr].mean()
    out = np.full(int((va_mask & P).sum()), g)
    ai = apt[va_mask & P]; bi = b[va_mask & P]
    for a in np.unique(ai):
        sa = ktr & (apt == a)
        ga = (tail[sa].sum() + prior_w * g) / (sa.sum() + prior_w) if sa.any() else g
        for bb in np.unique(bi):
            s = sa & (b == bb)
            v = (tail[s].sum() + prior_w * ga) / (s.sum() + prior_w) if s.any() else ga
            out[(ai == a) & (bi == bb)] = v
    return out


print('\n--- specialist classifiers (fold-safe, fitted on P-train only) ---', flush=True)
variants = {'spec': {}, 'spec_mono': {}, 'spec_mono3': {}, 'table': {}, 'iso': {}}
for m in MONTHS:
    va = month == m
    tr_ = ~va
    t = time.time()
    variants['spec'][m] = fit_spec(tr_, va)
    variants['spec_mono'][m] = fit_spec(tr_, va, mono=True)
    variants['spec_mono3'][m] = np.mean([fit_spec(tr_, va, mono=True, seed=s_)
                                         for s_ in (0, 1, 2)], axis=0)
    variants['table'][m] = table_est(tr_, va)
    variants['iso'][m] = iso_est(tr_, va)
    print(f'    month {m:02d}: {int((va & P).sum()):,} P-rows  ({time.time()-t:.0f}s)', flush=True)

ps = {k: np.full(len(y), np.nan) for k in variants}
for k, d in variants.items():
    for m, v in d.items():
        ps[k][(month == m) & P] = v

e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(os_[va]) & (os_[va] > TAIL_CAP), os_[va], tm)

base = mixture(pT, e_tail, rb)
preds = {'BASE global pT': base}
for k in variants:
    p2 = pT.copy()
    p2[P] = ps[k][P]
    preds[f'HYB {k}'] = mixture(p2, e_tail, rb)
# blends of the global and specialist scores
for w in [0.25, 0.5, 0.75]:
    p2 = pT.copy()
    p2[P] = (1 - w) * pT[P] + w * ps['spec'][P]
    preds[f'HYB blend w={w}'] = mixture(p2, e_tail, rb)
for a, b in [('spec', 'table'), ('spec_mono3', 'table'), ('spec_mono3', 'iso'),
             ('table', 'iso')]:
    p2 = pT.copy(); p2[P] = 0.5 * ps[a][P] + 0.5 * ps[b][P]
    preds[f'HYB {a}+{b}'] = mixture(p2, e_tail, rb)
p2 = pT.copy()
p2[P] = np.mean([ps[k][P] for k in ('spec_mono3', 'table', 'iso')], axis=0)
preds['HYB 3-way mean'] = mixture(p2, e_tail, rb)
# ceilings (not achievable -- they use the label)
p2 = pT.copy(); p2[P] = tail[P].astype(float)
preds['CEIL oracle p in P'] = mixture(p2, e_tail, rb)
p2 = pT.copy(); p2[P] = tail[P].mean()
preds['CEIL constant p in P'] = mixture(p2, e_tail, rb)

print('\n' + '=' * 118)
print(f'RESULTS, months {MONTHS}   ({int(sel.sum()):,} rows, {int(tail[sel].sum())} tail)')
print('=' * 118)
rows = []
for k, p in preds.items():
    d = decompose(y[sel], p[sel])
    s = sel & P
    rows.append(dict(model=k, rmse=d['rmse'], bulk=d['rmse_bulk'], tail=d['rmse_tail'],
                     c_tail=d['contrib_tail'],
                     inP_rmse=float(np.sqrt(np.mean((y[s] - p[s]) ** 2))),
                     inP_contrib=float(np.sqrt(((y[s] - p[s]) ** 2).sum() / sel.sum()))))
res = pd.DataFrame(rows)
print(res.round(2).to_string(index=False))

print('\n--- paired bootstrap vs BASE (negative = better) ---')
for k, p in preds.items():
    if k == 'BASE global pT':
        continue
    r = paired_bootstrap(y[sel], p[sel], base[sel])
    print(f'  {k:24s} {r["diff"]:+9.2f} +- {r["sd"]:6.2f}  CI [{r["lo"]:+8.1f}, '
          f'{r["hi"]:+8.1f}]  P(worse)={r["p_a_worse"]*100:5.1f}%')

print('\n--- classifier quality inside P (eval months only) ---')
s = sel & P
print(f'  n={int(s.sum()):,}  actual tail rate {tail[s].mean()*100:.2f}%')
for k, p in [('global pT', pT)] + [(kk, ps[kk]) for kk in variants]:
    v = p[s]; t_ = tail[s]
    ll = -np.mean(t_ * np.log(np.clip(v, 1e-9, 1)) + (1 - t_) * np.log(np.clip(1 - v, 1e-9, 1)))
    br = np.mean((v - t_) ** 2)
    ordr = np.argsort(-v)
    top = ordr[:int(t_.sum())]
    print(f'  {k:10s} mean_p={v.mean()*100:6.2f}%  logloss={ll:.4f}  brier={br:.4f}  '
          f'mean on tail={v[t_].mean()*100:6.2f}%  on bulk={v[~t_].mean()*100:6.2f}%  '
          f'P@k={t_[top].mean()*100:5.1f}%')

print('\n--- how the in-P error is distributed over off_sched (BASE vs best) ---')
bestk = res[~res.model.str.startswith('CEIL')].sort_values('rmse').iloc[0]['model']
edges = [7200, 9000, 12000, 20000, 40000, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    s_ = sel & P & (os_ >= lo) & (os_ < hi)
    if not s_.any():
        continue
    print(f'    off_sched [{lo:8.0f},{hi:10.0f}) n={int(s_.sum()):5,} '
          f'tail={int(tail[s_].sum()):4d}  mean pT={pT[s_].mean():.3f} '
          f'best_p={np.nan:.0f}'.replace('nan', '   ')
          + f'  BASE contrib={np.sqrt(((y[s_]-base[s_])**2).sum()/sel.sum()):7.1f}'
            f'  BEST contrib={np.sqrt(((y[s_]-preds[bestk][s_])**2).sum()/sel.sum()):7.1f}')
    for kk in ['spec_mono3', 'table', 'iso']:
        print(f'         mean {kk:11s} = {ps[kk][s_].mean():.3f}   '
              f'(actual rate {tail[s_].mean():.3f})')

print('\n--- per-month (best hybrid vs BASE) ---')
best = res[~res.model.str.startswith('CEIL')].sort_values('rmse').iloc[0]['model']
print(f'  best non-ceiling variant: {best}')
pm_b = per_month(y[sel], base[sel], month[sel])
pm_h = per_month(y[sel], preds[best][sel], month[sel])
cmp = pm_b[['month', 'n', 'n_tail', 'rmse']].rename(columns={'rmse': 'BASE'})
cmp['BEST'] = pm_h['rmse'].values
cmp['delta'] = cmp.BEST - cmp.BASE
print(cmp.round(2).to_string(index=False))

np.save(TAILDIR / 'exp36_preds.npy', np.vstack([preds[k] for k in preds]).astype(np.float32))
open(TAILDIR / 'exp36_keys.txt', 'w').write('\n'.join(preds))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
