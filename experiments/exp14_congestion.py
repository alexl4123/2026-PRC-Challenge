"""EXP14 -- do congestion features help? Measured under month-wise CV.

EXP13: July bulk RMSE 306.9 vs March 208.7, tail rows excluded. Summer
congestion is the leading explanation and nothing in the model captures it.

An earlier crude version of these features gave contradictory readings under the
old broken validation (helped in one setup, hurt in another). This run settles it
with 2.08M out-of-fold rows and a paired bootstrap.

Baseline and candidate are the SAME mixture architecture through the SAME folds;
the only difference is the 26 congestion columns.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS, NEEDED
from prc.congestion import build as build_cong
from prc.io import load, peak_rss_gb
from prc.validation import decompose, paired_bootstrap, month_cv_oof, per_month
pd.set_option('display.width', 210)

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
X = pd.concat([X, cong.set_axis(X.index)], axis=1)
CONG = list(cong.columns)
del cong
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)
print(f'{len(y):,} rows | {len(BASE)} base + {len(CONG)} congestion features '
      f'| {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)


def make_mixture(cols):
    cats = [c for c in CATS if c in cols]

    def fp(Xt, yt, Xv):
        k = yt <= TAIL_CAP
        mb = lgb.train(REG, lgb.Dataset(Xt[cols][k], yt[k], categorical_feature=cats),
                       num_boost_round=350)
        mc = lgb.train(CLF, lgb.Dataset(Xt[cols], (yt > TAIL_CAP).astype(int),
                                        categorical_feature=cats), num_boost_round=300)
        pb = np.clip(mb.predict(Xv[cols]), 0, None)
        pt = mc.predict(Xv[cols])
        osv = Xv['off_sched'].to_numpy()
        et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv,
                      yt[yt > TAIL_CAP].mean())
        return np.clip(pt * et + (1 - pt) * pb, 0, None)
    return fp


oof = {}
for name, cols in [('base', BASE), ('base+congestion', BASE + CONG)]:
    print(f'\n=== {name} ({len(cols)} features) ===', flush=True)
    oof[name] = month_cv_oof(X, y, month, make_mixture(cols))
    d = decompose(y, oof[name])
    print(f'  OOF RMSE {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} | '
          f'tail {d["rmse_tail"]:8.0f}', flush=True)

r = paired_bootstrap(y, oof['base+congestion'], oof['base'])
print('\n' + '=' * 90)
print(f'congestion vs base: {r["diff"]:+.2f} +- {r["sd"]:.2f}  '
      f'95% CI [{r["lo"]:+.1f}, {r["hi"]:+.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
print('\nper-month bulk RMSE, base -> base+congestion:')
a = per_month(y, oof['base'], month).set_index('month')
b = per_month(y, oof['base+congestion'], month).set_index('month')
cmp = pd.DataFrame({'base_bulk': a.rmse_bulk, 'cong_bulk': b.rmse_bulk})
cmp['delta'] = cmp.cong_bulk - cmp.base_bulk
print(cmp.round(1).to_string())
print(f'\n  JULY specifically: {cmp.loc[7,"base_bulk"]:.1f} -> {cmp.loc[7,"cong_bulk"]:.1f} '
      f'({cmp.loc[7,"delta"]:+.1f})')
print(f'  JANUARY          : {cmp.loc[1,"base_bulk"]:.1f} -> {cmp.loc[1,"cong_bulk"]:.1f} '
      f'({cmp.loc[1,"delta"]:+.1f})')
np.save('cache/exp14_oof.npy', np.vstack([oof['base'], oof['base+congestion']]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
