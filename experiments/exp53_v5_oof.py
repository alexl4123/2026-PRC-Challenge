"""EXP53 -- the OOF number for the v5 configuration, which was never measured.

v5 = v4 with one change: the bulk regressor goes from 255 leaves / 350 rounds /
lr 0.05 to 511 / 1400 / lr 0.025 (exp50's confirmed winner).

exp50 only measured that change on three individual months, BULK ROWS ONLY
(-2.06, -1.04, -1.69). This runs the full 12-fold leave-one-month-out OOF on
the complete mixture so v5 sits on the same scale as v3 (296.22) and v4
(287.81) and the comparison is like for like.

Single seed, matching the `r-full__c-base` single-seed figure of 289.34 from
the cluster grid rather than its 10-seed 287.81 -- a seed-averaged run would
cost 12x the compute for a ~1.5s effect we have already measured separately.
So the number to compare against here is 289.34, not 287.81.

~1 hour: 1400 rounds at 511 leaves is roughly 4x the old budget.
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
from prc.validation import decompose, paired_bootstrap, month_cv_oof, per_month
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
X, = align_categories(X)
print(f'{len(y):,} rows | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
RCAT = [c for c in CATS if c in FULL]
CCAT = [c for c in CATS if c in BASE]


def make(leaves, rounds, lr):
    reg = dict(objective='regression', metric='rmse', learning_rate=lr,
               num_leaves=leaves, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8,
               verbose=-1, seed=0, deterministic=True, force_row_wise=True)

    def fp(Xt, yt, Xv):
        k = yt <= TAIL_CAP
        mb = lgb.train(reg, lgb.Dataset(Xt[FULL][k], yt[k], categorical_feature=RCAT),
                       num_boost_round=rounds)
        mc = lgb.train(CLF, lgb.Dataset(Xt[BASE], (yt > TAIL_CAP).astype(int),
                                        categorical_feature=CCAT), num_boost_round=300)
        pb = np.clip(mb.predict(Xv[FULL]), 0, None)
        pt = mc.predict(Xv[BASE])
        osv = Xv['off_sched'].to_numpy()
        et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
        return np.clip(pt * et + (1 - pt) * pb, 0, None)
    return fp


oof = {}
for name, (lv, rd, lr) in [('v4 config  255/350/0.05', (255, 350, 0.05)),
                           ('v5 config 511/1400/0.025', (511, 1400, 0.025))]:
    print(f'\n=== {name} ===', flush=True)
    oof[name] = month_cv_oof(X, y, month, make(lv, rd, lr))
    d = decompose(y, oof[name])
    print(f'  OOF {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} | tail {d["rmse_tail"]:8.0f}',
          flush=True)

a, b = list(oof)
r = paired_bootstrap(y, oof[b], oof[a])
print('\n' + '=' * 84)
print(f'  v5 - v4 (single seed, same folds): {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
      f'CI [{r["lo"]:+6.2f}, {r["hi"]:+6.2f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
print('\n  per-month:')
t = pd.DataFrame({k.split()[0] + k.split()[1]: per_month(y, v, month).set_index('month').rmse
                  for k, v in oof.items()})
t['delta'] = t.iloc[:, 1] - t.iloc[:, 0]
print(t.round(2).to_string())
np.save('cache/exp53_oof.npy', np.vstack([oof[k] for k in oof]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
