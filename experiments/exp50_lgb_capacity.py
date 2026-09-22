"""EXP50 -- LightGBM capacity and regularisation, never tuned until now.

`num_leaves=255, num_boost_round=350, learning_rate=0.05, min_data_in_leaf=100`
was a first guess on day one and has been frozen through every experiment since.
Two reasons to suspect it is wrong:

  * the bulk regressor trains on 1.89M rows with 72 features, and 255 leaves is
    a modest tree for that;
  * the 480-task grid found 54 extra features bought only ~2s of bulk. That is
    the signature of a SATURATED model as much as of weak features, and the two
    are distinguishable: if capacity is the binding constraint, a bigger model
    should convert those same features into more.

Evidence against heavy tuning exists too -- a 2025 competitor's 160-trial Optuna
run produced their WORST submission -- so this is a coarse sweep of a handful of
sensible points, not a search.

METHOD. Only the BULK regressor is varied; the classifier and the tail anchor
are held fixed, and only bulk rows (y <= 7200) are scored. Mixing the tail in
would swamp a 2-5s bulk effect with ~150 rows of noise.

MONTH 4 IS THE SIZING MONTH, deliberately. Months 1, 7 and 10 are the ones we
report elsewhere, and choosing hyperparameters on them would be selection on the
evaluation set. Month 4 is typical (bulk 206.0). The best two configurations are
then CONFIRMED on months 2 and 9, which were used for nothing.
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
from prc.validation import rmse, paired_bootstrap
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
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)],
              axis=1)
del cong, wx, atfm
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)
COLS = list(X.columns)
CATF = [c for c in CATS if c in COLS]
print(f'{len(y):,} rows, {len(COLS)} features | {time.time()-t0:.0f}s | '
      f'RSS {peak_rss_gb():.2f} GB', flush=True)

BASE = dict(objective='regression', metric='rmse', feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1,
            seed=0, deterministic=True, force_row_wise=True)


def run(val_month, leaves, rounds, lr, mdl, l2):
    va = month == val_month
    trn = (~va) & (y <= TAIL_CAP)
    p = dict(BASE, num_leaves=leaves, learning_rate=lr, min_data_in_leaf=mdl,
             lambda_l2=l2)
    m = lgb.train(p, lgb.Dataset(X.loc[trn, COLS], y[trn], categorical_feature=CATF),
                  num_boost_round=rounds)
    ev = va & (y <= TAIL_CAP)
    return np.clip(m.predict(X.loc[ev, COLS]), 0, None), y[ev]


GRID = [
    #  leaves rounds   lr   min_leaf  l2     label
    (  255,   350,  0.05,   100,   1.0, 'current (frozen since day one)'),
    (  255,   700,  0.05,   100,   1.0, 'more rounds'),
    (  511,   350,  0.05,   100,   1.0, 'wider'),
    (  511,   700,  0.05,   100,   1.0, 'wider + more rounds'),
    ( 1023,   700,  0.05,   100,   1.0, 'much wider'),
    (  511,  1400,  0.025,  100,   1.0, 'wider, slower, longer'),
    (  511,   700,  0.05,    20,   1.0, 'wider, finer leaves'),
    (  511,   700,  0.05,   100,  10.0, 'wider, stronger L2'),
]
print('\n--- sizing on month 4 (never reported elsewhere) ---', flush=True)
res = {}
for leaves, rounds, lr, mdl, l2, label in GRID:
    t = time.time()
    p, yy = run(4, leaves, rounds, lr, mdl, l2)
    res[label] = (rmse(yy, p), p, yy, (leaves, rounds, lr, mdl, l2))
    print(f'  {label:32s} leaves={leaves:5d} rounds={rounds:5d} lr={lr:<6g} '
          f'bulk RMSE {res[label][0]:7.2f}   ({time.time()-t:.0f}s)', flush=True)

cur = res['current (frozen since day one)']
ranked = sorted(res, key=lambda k: res[k][0])
print(f'\n  best: {ranked[0]}  ({res[ranked[0]][0]:.2f} vs current {cur[0]:.2f}, '
      f'{res[ranked[0]][0]-cur[0]:+.2f})')
for k in ranked[:3]:
    if k == 'current (frozen since day one)':
        continue
    r = paired_bootstrap(cur[2], res[k][1], cur[1])
    print(f'    {k:32s} vs current: {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
          f'CI [{r["lo"]:+6.2f}, {r["hi"]:+6.2f}]')

print('\n--- CONFIRMING the best two on months 2 and 9 (used for nothing else) ---',
      flush=True)
best = [k for k in ranked if k != 'current (frozen since day one)'][:2]
for vm in (2, 9):
    pc, yy = run(vm, *cur[3])
    print(f'  month {vm}: current {rmse(yy, pc):7.2f}', flush=True)
    for k in best:
        p, _ = run(vm, *res[k][3])
        r = paired_bootstrap(yy, p, pc)
        print(f'           {k:32s} {rmse(yy,p):7.2f}  ({r["diff"]:+6.2f} +- {r["sd"]:5.2f}, '
              f'CI [{r["lo"]:+6.2f}, {r["hi"]:+6.2f}])', flush=True)
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
