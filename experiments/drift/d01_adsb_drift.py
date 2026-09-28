"""Label-free 2025 -> 2026 drift of the 35 adsb.lol columns (GREKI, Discord 2026-09-28: "features encoding where an
aircraft was heard didn't transfer; features describing its movement did"). Jan+Jul 2025 vs Jan+Jul 2026, per airport:
coverage (non-null share) and, among non-null values, the two-sample KS distance. Output: cache/drift/d01_*.csv.
    python experiments/drift/d01_adsb_drift.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from prc.config import CACHE, ID
from prc.io import load

COVERAGE = ['adsb_matched', 'adsb_match_kind', 'adsb_to_dev', 'adsb_to_obs', 'adsb_has_ground', 'adsb_best_src', 'adsb_gs_first',
            'adsb_n_pts', 'adsb_max_gap_s', 'adsb_cov_s_per_pt', 'adsb_mlat_frac', 'adsb_at_park', 'adsb_day_ok',
            'adsb_apt_day_ground', 'adsb_apt_day_best', 'adsb_d_stand_first', 'adsb_stand_known']
OUT = CACHE / 'drift'; OUT.mkdir(exist_ok=True)
frames = []
for w in ('training', 'ranking'):
    d = load(w, phase='DEP', columns=[ID, 'APT', 'MVT_TIME_UTC_mvt'])
    a = pd.read_parquet(CACHE / 'adsb_lol' / f'features_{w}.parquet')
    a = a.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(d[ID].to_numpy())
    a['apt'] = d.APT.astype(str).to_numpy(); a['month'] = d.MVT_TIME_UTC_mvt.dt.month.to_numpy(); a['year'] = 2025 if w == 'training' else 2026
    frames.append(a[a.month.isin([1, 7])])
A = pd.concat(frames, ignore_index=True)
cols = [c for c in A.columns if c.startswith('adsb_')]
rows = []
for c in cols:
    for apt in sorted(A.apt.unique()):
        x5 = A.loc[(A.apt == apt) & (A.year == 2025), c].astype(float); x6 = A.loc[(A.apt == apt) & (A.year == 2026), c].astype(float)
        c5, c6 = x5.notna().mean(), x6.notna().mean()
        v5, v6 = x5.dropna().to_numpy(), x6.dropna().to_numpy()
        ks = ks_2samp(v5, v6).statistic if min(len(v5), len(v6)) >= 200 else np.nan
        rows.append(dict(col=c, kind='coverage' if c in COVERAGE else 'movement', apt=apt, cov25=c5, cov26=c6, dcov=c6 - c5, ks=ks))
T = pd.DataFrame(rows); T.to_csv(OUT / 'd01_adsb_drift_by_airport.csv', index=False)
S = T.groupby(['kind', 'col']).agg(max_abs_dcov=('dcov', lambda v: np.nanmax(np.abs(v))), mean_ks=('ks', 'mean'), max_ks=('ks', 'max')).round(3)
S.to_csv(OUT / 'd01_adsb_drift_summary.csv')
pd.set_option('display.width', 200)
print(S.sort_values(['kind', 'mean_ks'], ascending=[True, False]).to_string())
print('\nmean KS among covered rows by kind:', T.groupby('kind').ks.mean().round(3).to_dict())
