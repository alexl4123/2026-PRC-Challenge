"""EXP16 -- give the bulk regressor and the tail classifier DIFFERENT features.

EXP15 found weather helps the bulk (July -8.0s, the largest single-month gain
measured) but hurts the tail, so the package nets out to nothing:

    config              OOF      bulk     tail
    base+cong         291.86    239.6     9963
    +weather          293.45    236.8    10361
    +weather+atfm     297.27    237.7    10668

The likely cause is that the extra 28 features went to BOTH components. The tail
classifier learns from 435 positives in 1.74M rows; handing it 28 more mostly
irrelevant columns makes it noisier, and the tail dominates the total.

So: feed the rich feature set to the bulk regressor, keep the classifier on the
lean set that was working. The tail anchor (off_sched) is unchanged.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from prc.config import TARGET, TAIL_CAP
from prc.features import build as build_base, align_categories, CATS, NEEDED
from prc.congestion import build as build_cong
from prc.weather import download, prepare, attach as attach_wx
from prc.disruption import attach as attach_atfm
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
wx = attach_wx(tr, prepare(download()))
atfm = attach_atfm(tr)
CONG, WX, ATFM = list(cong.columns), list(wx.columns), list(atfm.columns)
X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index), atfm.set_axis(X.index)],
              axis=1)
del cong, wx, atfm
y = tr[TARGET].astype(float).to_numpy()
month = tr.MVT_TIME_UTC_mvt.dt.month.to_numpy()
X, = align_categories(X)
print(f'{len(y):,} rows | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)


def make(reg_cols, clf_cols):
    rc = [c for c in CATS if c in reg_cols]
    cc = [c for c in CATS if c in clf_cols]

    def fp(Xt, yt, Xv):
        k = yt <= TAIL_CAP
        mb = lgb.train(REG, lgb.Dataset(Xt[reg_cols][k], yt[k], categorical_feature=rc),
                       num_boost_round=350)
        mc = lgb.train(CLF, lgb.Dataset(Xt[clf_cols], (yt > TAIL_CAP).astype(int),
                                        categorical_feature=cc), num_boost_round=300)
        pb = np.clip(mb.predict(Xv[reg_cols]), 0, None)
        pt = mc.predict(Xv[clf_cols])
        osv = Xv['off_sched'].to_numpy()
        et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
        return np.clip(pt * et + (1 - pt) * pb, 0, None)
    return fp


RICH = BASE + CONG + WX + ATFM
sets = [
    ('reg=base+cong   clf=base+cong', BASE + CONG, BASE + CONG),          # exp14 best
    ('reg=RICH        clf=base+cong', RICH, BASE + CONG),                 # the fix
    ('reg=RICH        clf=base',      RICH, BASE),                        # leaner still
]
oof = {}
for name, rc, cc in sets:
    print(f'\n=== {name} (reg {len(rc)}, clf {len(cc)}) ===', flush=True)
    oof[name] = month_cv_oof(X, y, month, make(rc, cc))
    d = decompose(y, oof[name])
    print(f'  OOF {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} | tail {d["rmse_tail"]:8.0f}',
          flush=True)

ref = sets[0][0]
print('\n' + '=' * 94)
for name, _, _ in sets[1:]:
    r = paired_bootstrap(y, oof[name], oof[ref])
    print(f'  {name} vs baseline: {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
          f'CI [{r["lo"]:+6.1f}, {r["hi"]:+6.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
print('\nper-month bulk RMSE:')
tab = pd.DataFrame({n: per_month(y, o, month).set_index('month').rmse_bulk
                    for n, o in oof.items()})
print(tab.round(1).to_string())
np.save('cache/exp16_oof.npy', np.vstack([oof[n] for n, _, _ in sets]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
