"""EXP38 -- can out-of-fold artifact propensities sharpen p inside P?

EXP36 established that the whole tail lives in P = {AOBT_3 missing, off_sched >
7200} (4,944 rows, 450 tail) and that a specialist fitted on P alone beats the
global classifier there. The remaining gap is large: in-P RMSE ~2,400 against an
oracle 1,394.

P has only 4,944 rows, so anything estimated INSIDE it is starved. But the
stamping artifact itself is not rare: A = {|BLOCK - SCHED| <= 60s} fires on
199,634 rows across the whole dataset. If a stand or an operator that gets
stamped at schedule in ordinary traffic is also the one that gets stamped at
schedule when the NM record is missing, then an A-rate estimated from 2M rows is
a far better per-group propensity than a tail rate estimated from 4,944.

This tests that: out-of-fold group rates of A, of aobt_missing, and of the
in-P tail itself, as extra numeric features for the specialist, plus a
(airport, pier) x off_sched table alongside the (airport) x off_sched one.

  /home/thinklex/prc2026-venv/bin/python experiments/exp38_propensity.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TAIL_CAP
from prc.features import CATS
from prc.io import peak_rss_gb
from prc.validation import decompose, paired_bootstrap, per_month
from prc.models.tail_harness import (assemble, run_component, mixture, params, CLF,
                                     fp_bulk_regressor, fp_tail_classifier, TAILDIR)
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture, OS_BUCKETS
pd.set_option('display.width', 230)

MONTHS = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
SEED, THREADS = 0, 4

t0 = time.time()
X, meta = assemble()
y = meta['y'].to_numpy(); month = meta['month'].to_numpy()
os_ = meta['off_sched'].to_numpy(); apt = meta['apt'].to_numpy()
miss = meta['aobt_missing'].to_numpy()
A = (np.abs(meta['delta'].to_numpy()) <= 60).astype(float)
tail = y > TAIL_CAP
P = population(os_, miss)
sel = np.isin(month, MONTHS)
def _s(col):
    # Categorical columns carry NaN for levels unseen in the fold's training
    # vocabulary; astype(str) leaves those as floats and np.char.add then dies.
    return col.astype('object').fillna('NA').astype(str).to_numpy()


pier = _s(X.PIER)
stand = _s(X.STAND_mvt)
oper = _s(X.AIRCRAFT_OPERATOR_flt)
keys = {'apt_pier': np.char.add(np.char.add(apt, '|'), pier),
        'apt_stand': np.char.add(np.char.add(apt, '|'), stand),
        'apt_oper': np.char.add(np.char.add(apt, '|'), oper)}
print(f'{sel.sum():,} eval rows, {int(tail[sel].sum())} tail; P={int(P.sum()):,} '
      f'({int(tail[P].sum())} tail)  ({time.time()-t0:.0f}s)', flush=True)

rb = run_component('bulk_reg', X, meta, fp_bulk_regressor, MONTHS, SEED, THREADS)
pT = run_component('clf_T', X, meta, fp_tail_classifier, MONTHS, SEED, THREADS)


def oof_rate(key, val, tr_mask, va_idx, prior_w, grand):
    """Beta-shrunk group mean of `val`, estimated on tr_mask only."""
    s = pd.Series(val[tr_mask]).groupby(key[tr_mask]).agg(['sum', 'size'])
    r = ((s['sum'] + prior_w * grand) / (s['size'] + prior_w)).to_dict()
    return np.array([r.get(k, grand) for k in key[va_idx]])


def table2(tr_mask, va_mask, prior_w=6.0):
    """(airport, pier) x off_sched bucket, shrunk pier -> airport -> global.

    Vectorised: the obvious per-row loop is O(n_val x 2.08M) because every mask
    scans the whole dataset, which for July alone is 1e10 operations.
    """
    b = np.digitize(os_, OS_BUCKETS) - 1
    ktr = tr_mask & P
    g = tail[ktr].mean()
    t_tr = pd.Series(tail[ktr].astype(float))
    ka, kp = apt[ktr], pier[ktr]
    ga = t_tr.groupby(ka).agg(['sum', 'size'])
    ga = ((ga['sum'] + prior_w * g) / (ga['size'] + prior_w))
    kpp = pd.Index(ka).str.cat(pd.Index(kp), sep='|')
    gp = t_tr.groupby(kpp.to_numpy()).agg(['sum', 'size'])
    gp_prior = pd.Series([ga.get(k.split('|')[0], g) for k in gp.index], index=gp.index)
    gp = ((gp['sum'] + prior_w * gp_prior) / (gp['size'] + prior_w))
    kb = kpp.str.cat(pd.Index(b[ktr].astype(str)), sep='|')
    gb = t_tr.groupby(kb.to_numpy()).agg(['sum', 'size'])
    gb_prior = pd.Series([gp.get(k.rsplit('|', 1)[0], g) for k in gb.index], index=gb.index)
    gb = ((gb['sum'] + prior_w * gb_prior) / (gb['size'] + prior_w))

    vi = va_mask & P
    va_a, va_p, va_b = apt[vi], pier[vi], b[vi].astype(str)
    kpp_v = pd.Index(va_a).str.cat(pd.Index(va_p), sep='|')
    kb_v = kpp_v.str.cat(pd.Index(va_b), sep='|')
    out = np.array([gb.get(kb_, gp.get(kp_, ga.get(ka_, g)))
                    for kb_, kp_, ka_ in zip(kb_v, kpp_v, va_a)], float)
    return out


SPEC = ['APT', 'PIER', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
        'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt',
        'off_sched', 'log_off_sched', 'hour', 'dow']
PROP = [f'{src}_{k}' for src in ('Arate', 'Mrate') for k in keys] + \
       [f'Trate_{k}' for k in ('apt_pier', 'apt_oper')]
SPEC_CATS = [c for c in CATS if c in SPEC]
SPEC_PAR = dict(CLF, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.9,
                bagging_fraction=0.9)

print('\n--- per-fold specialists ---', flush=True)
store = {k: np.full(len(y), np.nan) for k in
         ['base3', 'prop3', 'table1', 'table2', 'iso']}
from sklearn.isotonic import IsotonicRegression
for m in MONTHS:
    t = time.time()
    va = month == m
    tr_ = ~va
    vi = np.where(va & P)[0]
    Xp = X.loc[tr_ & P, SPEC].copy()
    Xv = X.loc[va & P, SPEC].copy()
    tp = tail[tr_ & P].astype(int)
    for k, key in keys.items():
        Xp[f'Arate_{k}'] = oof_rate(key, A, tr_, np.where(tr_ & P)[0], 20.0, A[tr_].mean())
        Xv[f'Arate_{k}'] = oof_rate(key, A, tr_, vi, 20.0, A[tr_].mean())
        Xp[f'Mrate_{k}'] = oof_rate(key, miss.astype(float), tr_, np.where(tr_ & P)[0],
                                    20.0, miss[tr_].mean())
        Xv[f'Mrate_{k}'] = oof_rate(key, miss.astype(float), tr_, vi, 20.0, miss[tr_].mean())
    for k in ('apt_pier', 'apt_oper'):
        gr = tail[tr_ & P].mean()
        Xp[f'Trate_{k}'] = oof_rate(keys[k], tail.astype(float), tr_ & P,
                                    np.where(tr_ & P)[0], 8.0, gr)
        Xv[f'Trate_{k}'] = oof_rate(keys[k], tail.astype(float), tr_ & P, vi, 8.0, gr)

    mono_b = [1 if c in ('off_sched', 'log_off_sched') else 0 for c in SPEC]
    mono_p = mono_b + [0] * len(PROP)
    store['base3'][va & P] = np.mean([
        lgb.train(dict(params(SPEC_PAR, s, THREADS), monotone_constraints=mono_b),
                  lgb.Dataset(Xp[SPEC], tp, categorical_feature=SPEC_CATS),
                  num_boost_round=200).predict(Xv[SPEC]) for s in (0, 1, 2)], axis=0)
    store['prop3'][va & P] = np.mean([
        lgb.train(dict(params(SPEC_PAR, s, THREADS), monotone_constraints=mono_p),
                  lgb.Dataset(Xp[SPEC + PROP], tp, categorical_feature=SPEC_CATS),
                  num_boost_round=200).predict(Xv[SPEC + PROP]) for s in (0, 1, 2)], axis=0)

    # table1 == the (airport x off_sched) table already in tail_specialist
    sp = TailSpecialist(threads=THREADS).fit(X[tr_], y[tr_], apt[tr_], os_[tr_], miss[tr_])
    ov = np.asarray(os_)[va & P]; av = np.asarray(apt)[va & P]
    store['table1'][va & P] = sp._table_predict(av, ov)
    store['iso'][va & P] = sp._iso_predict(av, ov)
    store['table2'][va & P] = table2(tr_, va)
    print(f'    month {m:02d}: {len(vi):,} P-rows  ({time.time()-t:.0f}s)', flush=True)

e_tail = np.full(len(y), np.nan)
for m in MONTHS:
    va = month == m
    tm = y[~va][y[~va] > TAIL_CAP].mean()
    e_tail[va] = np.where(np.isfinite(os_[va]) & (os_[va] > TAIL_CAP), os_[va], tm)

base = mixture(pT, e_tail, rb)
preds = {'BASE global pT': base}
for k in store:
    preds[f'S {k}'] = mixture(apply_to_mixture(pT, store[k], P), e_tail, rb)
for combo in [('base3', 'table1', 'iso'), ('prop3', 'table1', 'iso'),
              ('base3', 'table2', 'iso'), ('prop3', 'table2', 'iso'),
              ('base3', 'prop3', 'table1', 'table2', 'iso')]:
    v = np.nanmean(np.vstack([store[k] for k in combo]), axis=0)
    preds['S mean(' + '+'.join(combo) + ')'] = mixture(apply_to_mixture(pT, v, P), e_tail, rb)
o = pT.copy(); o[P] = tail[P].astype(float)
preds['CEIL oracle'] = mixture(o, e_tail, rb)

print('\n' + '=' * 118)
print(f'RESULTS, months {MONTHS}')
print('=' * 118)
rows = []
for k, p in preds.items():
    d = decompose(y[sel], p[sel]); s_ = sel & P
    rows.append(dict(model=k, rmse=d['rmse'], bulk=d['rmse_bulk'], tail=d['rmse_tail'],
                     inP=float(np.sqrt(np.mean((y[s_] - p[s_]) ** 2)))))
res = pd.DataFrame(rows).sort_values('rmse')
print(res.round(2).to_string(index=False))
print('\n--- paired bootstrap vs BASE ---')
for k, p in preds.items():
    if k == 'BASE global pT':
        continue
    r = paired_bootstrap(y[sel], p[sel], base[sel])
    print(f'  {k:42s} {r["diff"]:+9.2f} +- {r["sd"]:6.2f}  CI [{r["lo"]:+8.1f}, '
          f'{r["hi"]:+8.1f}]  P(worse)={r["p_a_worse"]*100:5.1f}%')

bestk = res[res.model != 'CEIL oracle'].iloc[0]['model']
print(f'\n--- best = {bestk}; paired vs the exp36 3-way mean ---')
ref = preds['S mean(base3+table1+iso)']
r = paired_bootstrap(y[sel], preds[bestk][sel], ref[sel])
print(f'  {r["diff"]:+9.2f} +- {r["sd"]:6.2f}  CI [{r["lo"]:+8.1f}, {r["hi"]:+8.1f}]  '
      f'P(worse)={r["p_a_worse"]*100:.1f}%')
print('\n--- per-month ---')
pb = per_month(y[sel], base[sel], month[sel])
ph = per_month(y[sel], preds[bestk][sel], month[sel])
c = pb[['month', 'n', 'n_tail', 'rmse']].rename(columns={'rmse': 'BASE'})
c['BEST'] = ph['rmse'].values; c['delta'] = c.BEST - c.BASE
print(c.round(2).to_string(index=False))
np.save(TAILDIR / 'exp38_preds.npy', np.vstack([preds[k] for k in preds]).astype(np.float32))
open(TAILDIR / 'exp38_keys.txt', 'w').write('\n'.join(preds))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
