"""EXP61 -- the four-way comparison on the TWO SCORED MONTHS.

WHY THIS EXPERIMENT, AND WHY LOCALLY
------------------------------------
The cluster study campaign is queued behind ~48,000 of our own `v2unit` array
tasks on the broadwell partition; 137 prc-study tasks sit PENDING (Resources).
The `routed` arm -- the largest identified lever -- is 15/36 complete and will
not finish soon. So the decisive measurement runs here instead, narrowed from
twelve folds to the two months that are actually scored.

FOUR ARMS
---------
  mix           the shipped v5 mixture                          (baseline)
  mix+spec      v5 with the tail specialist inside P            (= what v6 ships)
  routed        four disjoint routes by the NM proxy
  routed+spec   routed, with route D's classifier replaced by
                the specialist inside P                          (NEW -- untested)

The fourth arm is the synthesis nobody has measured. Route D already restricts
the classifier to a 2.14% tail rate instead of the global 0.028%, which is most
of what the specialist was invented to do -- so the two may well be REDUNDANT.
That is the question: does the specialist still pay once routing has already
concentrated the problem, or did we just find the same gain twice?

SCORED MONTHS ONLY
------------------
Jan and Jul 2026 are what the leaderboard scores. Training on the other eleven
months and validating on 1 and 7 mirrors that. It is NOT a substitute for the
twelve-fold protocol -- ~150 tail rows give a bootstrap sd of roughly 47s, so
only a large effect is detectable here. Treat this as a screen, and let the
cluster confirm.
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.models.routed import RoutedModel, route_of
from prc.models.tail_specialist import TailSpecialist, population, apply_to_mixture
from prc.validation import rmse, decompose, paired_bootstrap

THREADS = 4
MONTHS = (1, 7)
t0 = time.time()

full = load('training')
cong = build_cong(full)
tr = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
del full; gc.collect()
X = build_base(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
BASE = list(X.columns)
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
               atfm.set_axis(X.index)], axis=1)
del cong, wx, atfm; gc.collect()
FULL = list(X.columns)
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
proxy = (tr.MVT_TIME_UTC_mvt - tr.AOBT_3_flt).dt.total_seconds().to_numpy()
sched_off = (tr.MVT_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
apt = X['APT'].astype(str).to_numpy()
miss = X['aobt_missing'].to_numpy().astype(bool)
X, = align_categories(X)
del tr; gc.collect()

rt = route_of(proxy)
print(f'{len(y):,} rows | routes ' +
      ' '.join(f'{k}={int((rt==k).sum()):,}' for k in 'ABCD') +
      f' | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

RCAT = [c for c in CATS if c in FULL]
CCAT = [c for c in CATS if c in BASE]
REG = dict(objective='regression', metric='rmse', learning_rate=0.025, num_leaves=511,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=0,
           bagging_seed=1, feature_fraction_seed=2, deterministic=True,
           force_row_wise=True)
CLF = dict(REG, objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50)


def fit_mix(trm, vam, with_spec):
    """The shipped v5 mixture; optionally with the tail specialist inside P."""
    yt = y[trm]
    k = yt <= TAIL_CAP
    mb = lgb.train(REG, lgb.Dataset(X.loc[trm, FULL][k], yt[k], categorical_feature=RCAT),
                   num_boost_round=1400)
    mc = lgb.train(CLF, lgb.Dataset(X.loc[trm, BASE], (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CCAT), num_boost_round=300)
    pb = np.clip(mb.predict(X.loc[vam, FULL]), 0, None)
    pt = mc.predict(X.loc[vam, BASE])
    if with_spec:
        spec = TailSpecialist(threads=THREADS).fit(
            X.loc[trm], yt, apt[trm], sched_off[trm], miss[trm])
        p_spec, in_P = spec.predict(X.loc[vam], apt[vam], sched_off[vam], miss[vam])
        pt = apply_to_mixture(pt, p_spec, in_P)
    so = sched_off[vam]
    et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, yt[yt > TAIL_CAP].mean())
    return np.clip(pt * et + (1 - pt) * pb, 0, None)


def fit_routed(trm, vam, with_spec):
    m = RoutedModel(FULL, BASE, CATS, seed=0, threads=THREADS,
                    specialist=with_spec).fit(
        X.loc[trm], y[trm], proxy[trm], sched_off[trm])
    return m.predict(X.loc[vam], proxy[vam], sched_off[vam])


ARMS = [('mix', fit_mix, False), ('mix+spec', fit_mix, True),
        ('routed', fit_routed, False), ('routed+spec', fit_routed, True)]

preds, ys = {a: [] for a, _, _ in ARMS}, []
for m in MONTHS:
    va, trm = month == m, month != m
    ys.append(y[va])
    print(f'\n--- month {m}: train {trm.sum():,} / validate {va.sum():,} '
          f'({int((y[va] > TAIL_CAP).sum())} tail rows) ---', flush=True)
    for name, fn, sp in ARMS:
        t = time.time()
        p = fn(trm, va, sp)
        preds[name].append(p)
        d = decompose(y[va], p)
        print(f'  {name:12s} {d["rmse"]:7.2f}  bulk {d["rmse_bulk"]:6.2f}  '
              f'tail {d["rmse_tail"]:8.1f}  (contrib {d["contrib_bulk"]:6.2f} / '
              f'{d["contrib_tail"]:6.2f})  [{time.time()-t:.0f}s]', flush=True)
        gc.collect()

yy = np.concatenate(ys)
P = {a: np.concatenate(v) for a, v in preds.items()}
print(f'\n{"="*78}\nPOOLED over months {MONTHS} ({len(yy):,} rows, '
      f'{int((yy > TAIL_CAP).sum())} tail)\n{"="*78}')
for a in P:
    d = decompose(yy, P[a])
    print(f'  {a:12s} {d["rmse"]:7.2f}   bulk {d["rmse_bulk"]:6.2f}   '
          f'tail {d["rmse_tail"]:8.1f}')

print(f'\n  paired bootstrap, each arm vs `mix` (negative = better):')
for a in P:
    if a == 'mix':
        continue
    b = paired_bootstrap(yy, P[a], P['mix'])
    print(f'    {a:12s} {b["diff"]:+8.2f} +- {b["sd"]:5.2f}  '
          f'CI [{b["lo"]:+8.2f}, {b["hi"]:+8.2f}]  P(worse) {b["p_a_worse"]*100:5.1f}%')

print(f'\n  is the specialist redundant once routing has concentrated the problem?')
b = paired_bootstrap(yy, P['routed+spec'], P['routed'])
print(f'    routed+spec vs routed  {b["diff"]:+8.2f} +- {b["sd"]:5.2f}  '
      f'CI [{b["lo"]:+8.2f}, {b["hi"]:+8.2f}]  P(worse) {b["p_a_worse"]*100:5.1f}%')

np.save('cache/exp61_preds.npy', np.vstack([P[a] for a, _, _ in ARMS]))
np.save('cache/exp61_y.npy', yy)
np.save('cache/exp61_month.npy', np.concatenate([np.full(len(v), m)
                                                 for m, v in zip(MONTHS, ys)]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
