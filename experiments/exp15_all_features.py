"""EXP15 -- congestion + weather + ATFM disruption, all under month-wise CV.

EXP14 result to beat: congestion alone gave -4.27 +- 1.78 overall but made JULY
slightly WORSE (+1.3) while every other month improved 4-6s. So July's 98s bulk
gap is not congestion.

New hypothesis: July is hard because of summer convective weather. The ATFM file
shows July 2025 carried 186,066 minutes of weather delay against July 2026's
56,404. METAR gives thunderstorm, gust and visibility at each takeoff time;
the ATFM file gives daily disruption by cause.

Ablation so each block's contribution is separable.
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
print(f'{len(y):,} rows | base {len(BASE)} + cong {len(CONG)} + wx {len(WX)} + '
      f'atfm {len(ATFM)} | {time.time()-t0:.0f}s | RSS {peak_rss_gb():.2f} GB', flush=True)
print(f'  weather coverage: {X[WX[0]].notna().mean()*100:.1f}%   '
      f'atfm coverage: {X[ATFM[0]].notna().mean()*100:.1f}%', flush=True)

REG = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=255,
           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)
CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0,
           deterministic=True, force_row_wise=True)


def make(cols):
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
        et = np.where(np.isfinite(osv) & (osv > TAIL_CAP), osv, yt[yt > TAIL_CAP].mean())
        return np.clip(pt * et + (1 - pt) * pb, 0, None)
    return fp


sets = [('base+cong', BASE + CONG),
        ('base+cong+wx', BASE + CONG + WX),
        ('base+cong+wx+atfm', BASE + CONG + WX + ATFM)]
oof = {}
for name, cols in sets:
    print(f'\n=== {name} ({len(cols)}) ===', flush=True)
    oof[name] = month_cv_oof(X, y, month, make(cols))
    d = decompose(y, oof[name])
    print(f'  OOF {d["rmse"]:7.2f} | bulk {d["rmse_bulk"]:6.1f} | tail {d["rmse_tail"]:8.0f}',
          flush=True)

print('\n' + '=' * 92)
prev = None
for name, _ in sets:
    if prev:
        r = paired_bootstrap(y, oof[name], oof[prev])
        print(f'  {name:20s} vs {prev:20s}: {r["diff"]:+7.2f} +- {r["sd"]:5.2f}  '
              f'CI [{r["lo"]:+6.1f}, {r["hi"]:+6.1f}]  P(worse)={r["p_a_worse"]*100:.1f}%')
    prev = name

print('\nper-month bulk RMSE:')
tab = pd.DataFrame({n: per_month(y, o, month).set_index('month').rmse_bulk
                    for n, o in oof.items()})
tab['delta_total'] = tab.iloc[:, -1] - tab.iloc[:, 0]
print(tab.round(1).to_string())
print(f'\n  JULY: {tab.iloc[6,0]:.1f} -> {tab.iloc[6,-2]:.1f}   '
      f'JANUARY: {tab.iloc[0,0]:.1f} -> {tab.iloc[0,-2]:.1f}')
np.save('cache/exp15_oof.npy', np.vstack([oof[n] for n, _ in sets]))
print(f'\npeak RSS = {peak_rss_gb():.2f} GB  total {(time.time()-t0)/60:.1f} min')
