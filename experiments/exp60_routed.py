"""EXP60 -- does the routed residual model beat the global mixture?

Baseline: the shipped v5 mixture, OOF 289.33 (bulk 235.1, tail 10,078).
Candidate: prc/models/routed.py -- four disjoint routes by the NM proxy.

Screening on three months first (2, 7, 9): July because it carries ~69% of the
scored weight and is our worst month, plus two ordinary ones so a July-specific
effect is distinguishable from a general one. If it clears here, it goes to the
cluster for all twelve folds and ten seeds.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
from prc.io import load, peak_rss_gb
from prc.models.routed import RoutedModel, route_of
from prc.validation import rmse, decompose, paired_bootstrap
pd.set_option('display.width', 200)

t0 = time.time()
full = load('training')
cong = build_cong(full)
tr = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
del full
X = build_base(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
BASE = list(X.columns)
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)],
              axis=1)
del cong, wx, atfm
FULL = list(X.columns)
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
proxy = (tr.MVT_TIME_UTC_mvt - tr.AOBT_3_flt).dt.total_seconds().to_numpy()
sched_off = (tr.MVT_TIME_UTC_mvt - tr.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
X, = align_categories(X)
rt = route_of(proxy)
print(f'{len(y):,} rows | routes ' +
      ' '.join(f'{k}={int((rt==k).sum()):,}' for k in 'ABCD') +
      f' | {time.time()-t0:.0f}s', flush=True)

RCAT = [c for c in CATS if c in FULL]
CCAT = [c for c in CATS if c in BASE]
REG = dict(objective='regression', metric='rmse', learning_rate=0.025, num_leaves=511,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)


def baseline(trn, va):
    """The shipped v5 mixture, refit on this fold."""
    yt = y[trn]; k = yt <= TAIL_CAP
    mb = lgb.train(REG, lgb.Dataset(X.loc[trn, FULL][k], yt[k], categorical_feature=RCAT),
                   num_boost_round=1400)
    mc = lgb.train(CLF, lgb.Dataset(X.loc[trn, BASE], (yt > TAIL_CAP).astype(int),
                                    categorical_feature=CCAT), num_boost_round=300)
    pb = np.clip(mb.predict(X.loc[va, FULL]), 0, None)
    pt = mc.predict(X.loc[va, BASE])
    so = sched_off[va]
    et = np.where(np.isfinite(so) & (so > TAIL_CAP), so, yt[yt > TAIL_CAP].mean())
    return np.clip(pt * et + (1 - pt) * pb, 0, None)


MONTHS = [2, 7, 9]
res = {'baseline': [], 'routed': []}
truth, mon = [], []
for m in MONTHS:
    va, trn = month == m, month != m
    t = time.time()
    pb_ = baseline(trn, va)
    t1 = time.time()
    rm = RoutedModel(FULL, BASE, CATS, seed=0, threads=8).fit(
        X.loc[trn], y[trn], proxy[trn], sched_off[trn])
    pr_ = rm.predict(X.loc[va], proxy[va], sched_off[va])
    res['baseline'].append(pb_); res['routed'].append(pr_)
    truth.append(y[va]); mon.append(np.full(va.sum(), m))
    db, dr = decompose(y[va], pb_), decompose(y[va], pr_)
    print(f'  month {m}: baseline {db["rmse"]:7.2f} (bulk {db["rmse_bulk"]:6.1f} '
          f'tail {db["rmse_tail"]:8.0f})   routed {dr["rmse"]:7.2f} '
          f'(bulk {dr["rmse_bulk"]:6.1f} tail {dr["rmse_tail"]:8.0f})   '
          f'[{t1-t:.0f}s / {time.time()-t1:.0f}s]', flush=True)

yy = np.concatenate(truth)
pb_ = np.concatenate(res['baseline']); pr_ = np.concatenate(res['routed'])
db, dr = decompose(yy, pb_), decompose(yy, pr_)
print('\n' + '=' * 84)
print(f'  pooled baseline {db["rmse"]:7.2f}  bulk {db["rmse_bulk"]:6.1f}  tail {db["rmse_tail"]:8.0f}')
print(f'  pooled routed   {dr["rmse"]:7.2f}  bulk {dr["rmse_bulk"]:6.1f}  tail {dr["rmse_tail"]:8.0f}')
r = paired_bootstrap(yy, pr_, pb_)
print(f'\n  routed - baseline: {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
      f'CI [{r["lo"]:+6.1f}, {r["hi"]:+6.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
print('\n  per route, pooled:')
rtv = route_of(np.concatenate([proxy[month == m] for m in MONTHS]))
for k in 'ACD':
    s = rtv == k
    if s.sum():
        print(f'    {k}: n={int(s.sum()):8,}  baseline {rmse(yy[s],pb_[s]):9.1f}   '
              f'routed {rmse(yy[s],pr_[s]):9.1f}')
np.save('cache/exp60.npy', np.vstack([pb_, pr_, yy]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
