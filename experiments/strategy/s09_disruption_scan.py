"""s09 -- scan 2026 (Jan, Jul) for airport-days that look disrupted, using only provided,
label-free-for-departures signals, and see how our served v11 predictions moved on them.

Per airport-day: departures' route-A NM proxy median (off_aobt3), route-D share, labelled
ARRIVAL taxi-in median (provided 2026 labels), our served v11 mean.  Anomalies are relative to
the same airport's same calendar month in 2025.  The 2025 side gives the calibration: on 2025
airport-days, how does the day-level departure residual of v9~ relate to the same anomalies?
EHAM 2-7 Jan 2026 (the v10 event) is the known positive control.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS
from prc.io import peak_rss_gb

M = CACHE / 'major'
dc = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched', 'month', 'y']
d25 = pd.read_parquet(M / 'dep_training.parquet', columns=dc); d26 = pd.read_parquet(M / 'dep_ranking.parquet', columns=dc)
a25 = pd.read_parquet(M / 'arr_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'month', 'y'])
a26 = pd.read_parquet(M / 'arr_ranking.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'month', 'y'])
d25['pred'] = np.load(M / 'v9s_oof.npy')
sub = pd.read_parquet(OUTPUTS / 'understanding-accordion_v11.parquet')
d26 = d26.merge(sub.rename(columns={'TAXITIME_SEC_mvt': 'pred'}), on='MVT_ID_mvt', how='left')
assert d26.pred.notna().all()
d26 = d26[d26.month.isin([1, 7])]; a26 = a26[a26.month.isin([1, 7])]


def day_table(dep, arr):
    dep = dep.copy(); arr = arr.copy()
    dep['day'] = dep.MVT_TIME_UTC_mvt.dt.strftime('%m-%d'); arr['day'] = arr.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')
    dep['A'] = np.isfinite(dep.off_aobt3) & (dep.off_aobt3 > 0) & (dep.off_aobt3 <= 7200)
    dep['D'] = ~np.isfinite(dep.off_aobt3)
    dep['proxy'] = dep.off_aobt3.where(dep.A)
    g = dep.groupby(['APT', 'month', 'day'], observed=True)
    t = pd.DataFrame({'n': g.size(), 'dshare': g.D.mean(), 'proxy_med': g.proxy.median(),
                      'proxy_p90': g.proxy.quantile(0.9), 'pred_mean': g.pred.mean()})
    if 'y' in dep and dep.y.notna().any():
        t['y_mean'] = g.y.apply(lambda s: s[s <= 7200].mean())
        dep['r'] = (dep.y - dep.pred).where(dep.y <= 7200)
        t['resid_mean'] = g.r.mean()
    arr = arr[arr.y.between(0, 3600)]
    t['arr_med'] = arr.groupby(['APT', 'month', 'day'], observed=True).y.median()
    return t.reset_index()


t25 = day_table(d25, a25); t26 = day_table(d26, a26)
ref = t25.groupby(['APT', 'month'], observed=True)[['dshare', 'proxy_med', 'arr_med', 'pred_mean']].median().add_suffix('_ref').reset_index()
for t in (t25, t26):
    m = t.merge(ref, on=['APT', 'month'], how='left')
    for c in ['dshare', 'proxy_med', 'arr_med', 'pred_mean']:
        t[c + '_anom'] = m[c].to_numpy() - m[c + '_ref'].to_numpy()

print('=== 2025: does the day-level departure residual (y - v9~, bulk) track the day anomalies?  (1,200+ airport-days)')
k = t25.resid_mean.notna()
for c in ['proxy_med_anom', 'arr_med_anom', 'dshare_anom', 'pred_mean_anom']:
    print(f'  corr(resid_mean, {c:16s}) = {np.corrcoef(t25.resid_mean[k], t25[c][k].fillna(0))[0,1]:+.3f}   '
          f'Spearman {t25.resid_mean[k].rank().corr(t25[c][k].fillna(0).rank()):+.3f}')
top = t25[k].sort_values('proxy_med_anom', ascending=False).head(12)
print('\n2025 top-12 airport-days by NM-proxy anomaly: served (v9~) vs truth')
print(top[['APT', 'month', 'day', 'n', 'dshare', 'proxy_med', 'proxy_med_anom', 'arr_med_anom', 'y_mean', 'pred_mean', 'resid_mean']].round(1).to_string(index=False))

print('\n=== 2026 scored months: top-20 airport-days by NM-proxy anomaly (and arrival anomaly), served v11')
cols = ['APT', 'month', 'day', 'n', 'dshare', 'proxy_med', 'proxy_med_anom', 'proxy_p90', 'arr_med', 'arr_med_anom', 'pred_mean', 'pred_mean_anom']
print(t26.sort_values('proxy_med_anom', ascending=False).head(20)[cols].round(1).to_string(index=False))
print('\n=== 2026: top-15 airport-days by ARRIVAL taxi-in anomaly')
print(t26.sort_values('arr_med_anom', ascending=False).head(15)[cols].round(1).to_string(index=False))
# how much does the served prediction follow the proxy anomaly, 2025 vs 2026 (slope of pred_mean_anom on proxy_med_anom)
for nm, t in [('2025 (v9~ OOF)', t25), ('2026 (v11 served)', t26)]:
    kk = t.proxy_med_anom.notna() & t.pred_mean_anom.notna()
    b = np.polyfit(t.proxy_med_anom[kk], t.pred_mean_anom[kk], 1)[0]
    print(f'{nm}: slope of day-mean prediction anomaly on NM-proxy anomaly = {b:.3f}')
kk = t25.proxy_med_anom.notna() & t25.y_mean.notna()
y_anom = t25.y_mean - t25.merge(t25.groupby(['APT', 'month'], observed=True).y_mean.median().rename('yref').reset_index(), on=['APT', 'month']).yref.to_numpy()
print(f'2025 truth: slope of day-mean y anomaly on NM-proxy anomaly = {np.polyfit(t25.proxy_med_anom[kk], y_anom[kk], 1)[0]:.3f}')
t26.to_csv(CACHE / 'strategy' / 's09_days2026.csv', index=False); t25.to_csv(CACHE / 'strategy' / 's09_days2025.csv', index=False)
print('peak', round(peak_rss_gb(), 2), 'GB')
