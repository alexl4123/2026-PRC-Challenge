"""EXP11 -- build v3 from the two-stage mixture (EXP08).

Local holdout, uncapped, paired against our submitted models:
    v1 trimmed    563.32   (official 543.67)
    v2 untrimmed  465.50   (official 507.55)
    mixture       397.26   -- paired vs v2: -66.1 +- 28.9, 95% CI [-131, -17]

Structure: E[y|x] = P(T|x)*E[y|T,x] + (1-P(T|x))*E[y|~T,x]
  E[y|~T,x] regressor on bulk rows only (best bulk RMSE)
  E[y| T,x] off_sched where off_sched > TAIL_CAP, else the tail mean
  P(T|x)    LightGBM binary, verified calibrated (ratio 0.99 on holdout)

Final models are refit on all twelve months with round counts scaled by the
extra data, then predictions are written for the ranking set.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP, ID
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.submit import build as build_submission
from prc.validation import rmse, decompose, paired_bootstrap

t0 = time.time()
tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
rk = load('ranking', phase='DEP', columns=NEEDED + [ID])


def enrich(df, X):
    X['aobt_missing'] = df.AOBT_3_flt.isna().astype('int8')
    X['lobt_missing'] = df.LOBT_flt.isna().astype('int8')
    X['off_lobt'] = (df.MVT_TIME_UTC_mvt - df.LOBT_flt).dt.total_seconds()
    X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
    X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
    return X


Xtr, Xrk = enrich(tr, build(tr)), enrich(rk, build(rk))
Xtr, Xrk = align_categories(Xtr, Xrk)
COLS = list(Xtr.columns)
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
os_tr = Xtr.off_sched.to_numpy()
os_rk = Xrk.off_sched.to_numpy()
print(f'loaded {time.time()-t0:.0f}s')

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)

# ---------------- holdout check (so v3 has a local number on record) ----------
val = np.isin(month, [1, 7]); trn = ~val
yv = y[val]
mb = lgb.train(REG, lgb.Dataset(Xtr.loc[trn & (y <= TAIL_CAP), COLS],
                                y[trn & (y <= TAIL_CAP)], categorical_feature=CATS),
               num_boost_round=350)
mc = lgb.train(CLF, lgb.Dataset(Xtr.loc[trn, COLS], (y[trn] > TAIL_CAP).astype(int),
                                categorical_feature=CATS), num_boost_round=300)
ma = lgb.train(REG, lgb.Dataset(Xtr.loc[trn, COLS], y[trn], categorical_feature=CATS),
               num_boost_round=350)
tail_mean = y[trn & (y > TAIL_CAP)].mean()
pb, pt = np.clip(mb.predict(Xtr.loc[val, COLS]), 0, None), mc.predict(Xtr.loc[val, COLS])
osv = os_tr[val]
et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, tail_mean)
mix = np.clip(pt * et + (1 - pt) * pb, 0, None)
pa = np.clip(ma.predict(Xtr.loc[val, COLS]), 0, None)
d = decompose(yv, mix)
print(f'\nholdout mixture RMSE {d["rmse"]:.2f}  (bulk {d["rmse_bulk"]:.1f}, '
      f'tail {d["rmse_tail"]:.0f})')
r = paired_bootstrap(yv, mix, pa)
print(f'  paired vs untrimmed(v2-style): {r["diff"]:+.2f} +- {r["sd"]:.2f}  '
      f'95% CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')

# ---------------- refit on all twelve months, predict ranking -----------------
scale = len(y) / trn.sum()
bulk_all = y <= TAIL_CAP
MB = lgb.train(REG, lgb.Dataset(Xtr.loc[bulk_all, COLS], y[bulk_all],
                                categorical_feature=CATS),
               num_boost_round=int(350 * scale))
MC = lgb.train(CLF, lgb.Dataset(Xtr[COLS], (y > TAIL_CAP).astype(int),
                                categorical_feature=CATS), num_boost_round=int(300 * scale))
TM = y[y > TAIL_CAP].mean()
pb_r = np.clip(MB.predict(Xrk[COLS]), 0, None)
pt_r = MC.predict(Xrk[COLS])
et_r = np.where(np.isfinite(os_rk) & (os_rk > TAIL_CAP), os_rk, TM)
pred = np.clip(pt_r * et_r + (1 - pt_r) * pb_r, 0, None)
print(f'\nranking: P(tail) mean {pt_r.mean():.6f} (train rate {(y>TAIL_CAP).mean():.6f}); '
      f'rows with p>0.5: {int((pt_r>0.5).sum())}')
print(f'pred mean {pred.mean():.1f}  median {np.median(pred):.1f}  max {pred.max():.0f}')
build_submission(rk[ID].to_numpy(), pred, version=3, clip=(0.0, 200000.0))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
