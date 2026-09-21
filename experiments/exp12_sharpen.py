"""EXP12 -- two known inefficiencies in the v3 mixture.

(a) BULK POLLUTION. The mixture drops bulk RMSE from 255.7 to 277.3 because a
    small non-zero P(T|x) drags 344k ordinary rows toward a huge tail estimate.
    Gating p to zero below a threshold should recover most of that.

(b) WEAK TAIL ESTIMATE. E[y|T,x] = off_sched scores 7,853 on true tail rows, but
    the mixture only realises 13,682. Part is the classifier, part is that
    off_sched is used raw. Try a fitted tail regressor instead of the raw anchor.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build, align_categories, CATS, NEEDED
from prc.io import load, peak_rss_gb
from prc.validation import rmse, decompose, paired_bootstrap

t0 = time.time()
tr = load('training', phase='DEP', columns=NEEDED + [TARGET])
X = build(tr)
X['aobt_missing'] = tr.AOBT_3_flt.isna().astype('int8')
X['lobt_missing'] = tr.LOBT_flt.isna().astype('int8')
X['off_lobt'] = (tr.MVT_TIME_UTC_mvt - tr.LOBT_flt).dt.total_seconds()
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
osd = X.off_sched.to_numpy()
X, = align_categories(X)
COLS = list(X.columns)
val = np.isin(month, [1, 7]); trn = ~val
yv, osv = y[val], osd[val]

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)

mb = lgb.train(REG, lgb.Dataset(X.loc[trn & (y <= TAIL_CAP), COLS],
                                y[trn & (y <= TAIL_CAP)], categorical_feature=CATS),
               num_boost_round=350)
mc = lgb.train(CLF, lgb.Dataset(X.loc[trn, COLS], (y[trn] > TAIL_CAP).astype(int),
                                categorical_feature=CATS), num_boost_round=300)
pb = np.clip(mb.predict(X.loc[val, COLS]), 0, None)
pt = mc.predict(X.loc[val, COLS])
tail_mean = y[trn & (y > TAIL_CAP)].mean()
et_raw = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, tail_mean)

base = np.clip(pt * et_raw + (1 - pt) * pb, 0, None)
d = decompose(yv, base); print(f'v3 baseline           RMSE {d["rmse"]:7.2f}  '
                              f'bulk {d["rmse_bulk"]:6.1f}  tail {d["rmse_tail"]:8.0f}')

print('\n--- (a) gate p to zero below a threshold ---')
best = ('none', d['rmse'], base)
for g in [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    pg = np.where(pt < g, 0.0, pt)
    m = np.clip(pg * et_raw + (1 - pg) * pb, 0, None)
    dd = decompose(yv, m)
    print(f'  gate p<{g:<8g} n_zeroed={int((pt<g).sum()):7,}  RMSE {dd["rmse"]:7.2f}  '
          f'bulk {dd["rmse_bulk"]:6.1f}  tail {dd["rmse_tail"]:8.0f}')
    if dd['rmse'] < best[1]:
        best = (f'gate {g:g}', dd['rmse'], m)

print('\n--- (b) fitted tail regressor instead of the raw off_sched anchor ---')
tail_trn = trn & (y > TAIL_CAP)
mt = lgb.train(dict(REG, num_leaves=15, min_data_in_leaf=5, learning_rate=0.05),
               lgb.Dataset(X.loc[tail_trn, COLS], y[tail_trn], categorical_feature=CATS),
               num_boost_round=120)
et_fit = np.clip(mt.predict(X.loc[val, COLS]), 0, None)
tv = yv > TAIL_CAP
print(f'  on TRUE tail rows: raw off_sched {rmse(yv[tv], et_raw[tv]):8.0f}   '
      f'fitted {rmse(yv[tv], et_fit[tv]):8.0f}   '
      f'blend {rmse(yv[tv], 0.5*et_raw[tv]+0.5*et_fit[tv]):8.0f}')
for nm, et in [('raw', et_raw), ('fitted', et_fit), ('blend', 0.5*et_raw + 0.5*et_fit)]:
    gbest = None
    for g in [0, 1e-4, 1e-3, 3e-3]:
        pg = np.where(pt < g, 0.0, pt)
        m = np.clip(pg * et + (1 - pg) * pb, 0, None)
        r = rmse(yv, m)
        if gbest is None or r < gbest[0]:
            gbest = (r, g, m)
    print(f'  E[y|T] = {nm:7s} best gate {gbest[1]:<7g} RMSE {gbest[0]:7.2f}')
    if gbest[0] < best[1]:
        best = (f'{nm}+gate{gbest[1]:g}', gbest[0], gbest[2])

print(f'\nBEST: {best[0]}  RMSE {best[1]:.2f}')
r = paired_bootstrap(yv, best[2], base)
print(f'  paired vs v3 baseline: {r["diff"]:+.2f} +- {r["sd"]:.2f}  '
      f'95% CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {time.time()-t0:.0f}s')
