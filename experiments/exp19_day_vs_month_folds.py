"""EXP19 -- which CV protocol actually predicts the leaderboard?

We now have ground truth for one change. Going from v3's configuration (base
18 features for both mixture components) to v4's (full 72-feature regressor,
lean classifier) moved the OFFICIAL score by -30.02s, while month-wise OOF
predicted -8.41s.

Two candidate explanations, and they imply different fixes:
  (a) leave-one-month-out removes the held-out month's SEASON from training
      entirely, so it cannot see the value of weather/disruption features. The
      real task keeps the season -- training has Jan 2025 and Jul 2025 when
      predicting Jan/Jul 2026.
  (b) something else (seed averaging, or 2026-specific tail behaviour).

The user's suggestion: hold out random DAYS instead of whole months, so every
season stays represented in training. That fixes (a) but removes the time gap,
since a held-out day sits between two training days.

TEST: run the SAME two arms through the SAME number of folds under both
grouping schemes, and see which protocol's delta is closer to the official
-30.02s. Six folds either way so the comparison is not confounded by fold count.

  DAY-GROUPED   : 365 days shuffled into 6 folds. Season present in training,
                  no time gap. Expected to be OPTIMISTIC in absolute terms.
  MONTH-GROUPED : 12 months in 6 consecutive pairs. Season absent from
                  training. Expected to be PESSIMISTIC about seasonal features.

Single seed throughout, so this isolates the FEATURE effect from the seed
averaging that also differed between v3 and v4.
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
from prc.validation import decompose, paired_bootstrap
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
doy = tr.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)

# --- fold assignments, six folds either way -----------------------------------
rng = np.random.default_rng(0)
days = np.unique(doy)
rng.shuffle(days)
day_fold_of = {d: i % 6 for i, d in enumerate(days)}
fold_day = np.array([day_fold_of[d] for d in doy])
fold_month = (month - 1) // 2          # Jan+Feb, Mar+Apr, ... six pairs
print(f'{len(y):,} rows | base {len(BASE)} | full {len(FULL)} | {time.time()-t0:.0f}s')
print(f'  day folds   : sizes {np.bincount(fold_day)}')
print(f'  month folds : sizes {np.bincount(fold_month)}', flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=5, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(REG, objective='binary', num_leaves=63, min_data_in_leaf=50)
CLF['metric'] = 'auc'


def oof(fold, reg_cols):
    p = np.full(len(y), np.nan)
    rc = [c for c in CATS if c in reg_cols]
    cc = [c for c in CATS if c in BASE]
    for f in range(6):
        va, trn = fold == f, fold != f
        yt = y[trn]
        k = yt <= TAIL_CAP
        mb = lgb.train(REG, lgb.Dataset(X.loc[trn, reg_cols][k], yt[k],
                                        categorical_feature=rc), num_boost_round=350)
        mc = lgb.train(CLF, lgb.Dataset(X.loc[trn, BASE], (yt > TAIL_CAP).astype(int),
                                        categorical_feature=cc), num_boost_round=300)
        pb = np.clip(mb.predict(X.loc[va, reg_cols]), 0, None)
        pt = mc.predict(X.loc[va, BASE])
        osv = X.loc[va, 'off_sched'].to_numpy()
        et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
        p[va] = np.clip(pt * et + (1 - pt) * pb, 0, None)
    return p


results = {}
for pname, fold in [('DAY-grouped', fold_day), ('MONTH-grouped', fold_month)]:
    for aname, cols in [('base-feats (v3-like)', BASE), ('full-feats (v4-like)', FULL)]:
        t = time.time()
        p = oof(fold, cols)
        results[(pname, aname)] = p
        d = decompose(y, p)
        print(f'  {pname:14s} {aname:22s} RMSE {d["rmse"]:7.2f}  bulk {d["rmse_bulk"]:6.1f}  '
              f'tail {d["rmse_tail"]:8.0f}   ({time.time()-t:.0f}s)', flush=True)

print('\n' + '=' * 92)
print('  DELTA from base-feats to full-feats, by protocol.')
print('  Ground truth: the official v3 -> v4 delta was -30.02s.')
print('=' * 92)
for pname in ['DAY-grouped', 'MONTH-grouped']:
    a = results[(pname, 'full-feats (v4-like)')]
    b = results[(pname, 'base-feats (v3-like)')]
    r = paired_bootstrap(y, a, b)
    print(f'  {pname:14s} {r["diff"]:+8.2f} +- {r["sd"]:5.2f}  '
          f'CI [{r["lo"]:+7.1f}, {r["hi"]:+7.1f}]')
np.save('cache/exp19_oof.npy', np.vstack([results[k] for k in results]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
