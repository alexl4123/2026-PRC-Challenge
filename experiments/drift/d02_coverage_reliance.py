"""How much do the members rely on the adsb.lol COVERAGE ("where heard") columns, and would 2026 predictions move
without them? (Exploratory MEASUREMENT for the drift audit, 2026-09-28.)
The 'mix' member's bulk regressor (cluster/study_task.reg_params, 1400 rounds) on a fixed 35% subsample of the 2025
training rows, two arms: ALL columns (the adsb2 set) vs NOCOV (minus the 17 coverage columns of d01).
  (1) 2025 cost: fits with January or July held out; bulk RMSE on that month's bulk rows (y <= 7200).
  (2) 2026 movement: fits on all 2025 (same subsample), predictions for the ranking set; mean and RMS of
      (NOCOV - ALL) per airport x month, next to the coverage shift.
  (3) gain importance share of the adsb columns in the ALL fit.
    python experiments/drift/d02_coverage_reliance.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE, TAIL_CAP
from prc.io import peak_rss_gb
from cluster.study_task import reg_params
from cluster.w2_features import assemble_w2_pair, columns, W2_CATS

COV = ['adsb_matched', 'adsb_match_kind', 'adsb_to_dev', 'adsb_to_obs', 'adsb_has_ground', 'adsb_best_src', 'adsb_gs_first',
       'adsb_n_pts', 'adsb_max_gap_s', 'adsb_cov_s_per_pt', 'adsb_mlat_frac', 'adsb_at_park', 'adsb_day_ok',
       'adsb_apt_day_ground', 'adsb_apt_day_best', 'adsb_d_stand_first', 'adsb_stand_known']
t0 = time.time()
Xtr, mtr, Xrk, mrk, groups = assemble_w2_pair(adsb=True, extra=True)
cols, _ = columns(groups); cats = [c for c in W2_CATS if c in cols]
nocov = [c for c in cols if c not in COV]
y = mtr['TAXITIME_SEC_mvt'].to_numpy(float) if 'TAXITIME_SEC_mvt' in mtr else mtr.iloc[:, -1].to_numpy(float)
month = mtr['month'].to_numpy()
print(f'assembled train {Xtr.shape} ranking {Xrk.shape}, {len(cols)} cols ({len(cols) - len(nocov)} coverage) | {time.time() - t0:.0f}s, RSS {peak_rss_gb():.1f} GB', flush=True)
sub = np.random.default_rng(0).random(len(y)) < 0.35
bulk = np.isfinite(y) & (y <= TAIL_CAP)
P = reg_params(0, 6)

def fit(use, trn):
    k = trn & sub & bulk
    return lgb.train(P, lgb.Dataset(Xtr.loc[k, use], y[k], categorical_feature=[c for c in cats if c in use]), num_boost_round=1400)

rows = []
for m in (1, 7):
    va = (month == m) & bulk
    res = {}
    for arm, use in (('all', cols), ('nocov', nocov)):
        md = fit(use, month != m)
        pr = np.clip(md.predict(Xtr.loc[va, use]), 0, None)
        res[arm] = np.sqrt(np.mean((y[va] - pr) ** 2))
        if arm == 'all' and m == 1:
            imp = pd.Series(md.feature_importance('gain'), index=use); imp = imp / imp.sum()
    print(f'2025 month {m} held out, bulk RMSE: all {res["all"]:.2f}  nocov {res["nocov"]:.2f}  ({res["nocov"] - res["all"]:+.2f}) | {time.time() - t0:.0f}s', flush=True)
ad = imp[[c for c in imp.index if c.startswith('adsb_')]]
print(f'gain share: all adsb columns {ad.sum():.3f}, coverage columns {imp[[c for c in COV if c in imp.index]].sum():.3f}; top coverage: '
      + ', '.join(f'{c} {v:.3f}' for c, v in imp[[c for c in COV if c in imp.index]].sort_values(ascending=False).head(6).items()))
pa = {}
for arm, use in (('all', cols), ('nocov', nocov)):
    md = fit(use, np.ones(len(y), bool))
    pa[arm] = np.clip(md.predict(Xrk[use]), 0, None)
d = pa['nocov'] - pa['all']
t = pd.DataFrame({'apt': Xrk['APT'].astype(str).to_numpy(), 'm': mrk['month'].to_numpy(), 'd': d})
g = t.groupby(['apt', 'm']).d.agg(mean='mean', rms=lambda v: np.sqrt(np.mean(v ** 2))).unstack('m').round(1)
print('2026 ranking: NOCOV - ALL bulk prediction, by airport x month (mean / RMS, s)'); print(g.to_string())
print(f'overall RMS {np.sqrt(np.mean(d ** 2)):.1f} s | {time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB')
