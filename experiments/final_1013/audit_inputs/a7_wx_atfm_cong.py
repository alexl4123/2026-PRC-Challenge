"""METAR / ATFM / congestion / w2_extra / base features: per-month coverage, 2025 (1,2,6,7) vs final 2026."""
import os, sys, gc
os.environ['PRC_RANKING'] = 'final'
P = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, P)
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 3000); pd.set_option('display.max_columns', 60)
from prc import weather, disruption, congestion, features
from prc.config import CACHE
assert str(CACHE).endswith('cache_final')
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_inputs/'
out = open(O + 'a7.out', 'w')
MON = ['25-01', '25-02', '25-06', '25-07', '26-01', '26-02', '26-06', '26-07']

# ---------- raw METAR per station-day
raw = weather.download()          # cached parquet only (no network if files exist)
obs = weather.prepare(raw)
obs['day'] = obs.t.dt.strftime('%Y-%m-%d'); obs['ym'] = obs.t.dt.strftime('%y-%m')
o = obs[obs.ym.isin(MON)]
cnt = o.groupby(['station', 'ym', 'day']).size().rename('n').reset_index()
print('METAR obs per station-day: median by station/month', file=out)
print(cnt.groupby(['station', 'ym']).n.median().unstack().to_string(), file=out)
print('days present per station/month', file=out)
print(cnt.groupby(['station', 'ym']).size().unstack().to_string(), file=out)
print('station-days with < 40 obs:', file=out)
print(cnt[cnt.n < 40].to_string(), file=out)
# max gap between consecutive obs per station within months
o = o.sort_values(['station', 't'])
o['gap'] = o.groupby('station').t.diff().dt.total_seconds() / 60
big = o[o.gap > 90][['station', 't', 'gap']]
print('obs gaps > 90 min in these months:', file=out); print(big.to_string(), file=out)
print('last obs per station', obs.groupby('station').t.max().to_dict(), file=out)
# field missing share per station/month
for c in ['tmpc', 'dwpc', 'drct', 'sknt', 'vsby', 'skyl1']:
    print(f'METAR {c} missing share by station/month', file=out)
    print(o[c].isna().groupby([o.station, o.ym]).mean().unstack().round(3).to_string(), file=out)
out.flush()

# ---------- ATFM coverage
d = disruption.load_daily()
d['ym'] = pd.to_datetime(d.date).dt.strftime('%y-%m')
print('\nATFM rows per apt/month (days)', file=out)
print(d[d.ym.isin(MON)].groupby(['apt', 'ym']).size().unstack().to_string(), file=out)
print('ATFM date range', d.date.min(), d.date.max(), file=out)
print('ATFM dly_total median/mean by apt-month', file=out)
print(d[d.ym.isin(MON)].groupby(['apt', 'ym']).dly_total.mean().unstack().round(0).to_string(), file=out)
print('ATFM flt_arr mean by apt-month', file=out)
print(d[d.ym.isin(MON)].groupby(['apt', 'ym']).flt_arr.mean().unstack().round(0).to_string(), file=out)
out.flush()


def frame(which):
    from prc.io import load, _downcast
    if which == 'ranking':
        return load(which)
    T = lambda s: pd.Timestamp(s, tz='UTC')
    # keep the last day before each block so trailing windows are not truncated
    flt = [[('MVT_TIME_UTC_mvt', '>=', T('2025-01-01')), ('MVT_TIME_UTC_mvt', '<', T('2025-03-01'))],
           [('MVT_TIME_UTC_mvt', '>=', T('2025-05-31')), ('MVT_TIME_UTC_mvt', '<', T('2025-08-01'))]]
    full = pd.read_parquet(CACHE / 'training.parquet', filters=flt).reset_index(drop=True)
    full = _downcast(full)
    for c in full.columns:
        if str(full[c].dtype) == 'category':
            full[c] = full[c].cat.remove_unused_categories()
    return full


def summarize(name, X, ym, apt, out):
    rows = []
    for c in X.columns:
        x = pd.to_numeric(X[c], errors='coerce')
        g = x.groupby(ym)
        q = g.quantile([.1, .5, .9]).unstack()
        rows.append(pd.DataFrame({'col': c, 'ym': q.index, 'miss': x.isna().groupby(ym).mean().reindex(q.index).values,
                                  'p10': q[.1].values, 'p50': q[.5].values, 'p90': q[.9].values}))
    t = pd.concat(rows)
    for v in ['miss', 'p10', 'p50', 'p90']:
        print(f'\n== {name} {v} ==', file=out)
        print(t.pivot(index='col', columns='ym', values=v).reindex(columns=MON).round(3 if v == 'miss' else 1).to_string(), file=out)
    fl = []
    for c in X.columns:
        tb = X[c].isna().groupby([apt, ym]).mean().unstack().reindex(columns=MON)
        for a_ in tb.index:
            for mo in ['01', '02', '06', '07']:
                a, b = tb.loc[a_, '26-' + mo], tb.loc[a_, '25-' + mo]
                oth = np.nanmean([tb.loc[a_, '26-' + q] for q in ['01', '02', '06', '07'] if q != mo])
                if a - b > 0.03 and a - oth > 0.03:
                    fl.append((c, a_, '26-' + mo, round(b, 3), round(a, 3), round(oth, 3)))
    print(f'{name} APT MISSING FLAGS (col, apt, month, 2025, 2026, other2026):', file=out)
    for f in fl:
        print('   ', f, file=out)


res = {}
for which in ['training', 'ranking']:
    full = frame(which)
    cong = congestion.build(full)
    dep = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    del full; gc.collect()
    X = features.build(dep)
    wx = weather.attach(dep, obs.drop(columns=['day', 'ym']))
    at = disruption.attach(dep)
    from cluster import w2_extra
    ex = w2_extra.build(dep, 'ranking')        # 'ranking' mode only READS the cached training wtc map
    ym = dep.MVT_TIME_UTC_mvt.dt.strftime('%y-%m'); apt = dep.APT.astype(str)
    day = dep.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    keep = ym.isin(MON).to_numpy()
    num = X[features.NUMS]
    allf = pd.concat([num, cong, wx, at, ex], axis=1)[keep].reset_index(drop=True)
    res[which] = (allf, ym[keep].reset_index(drop=True), apt[keep].reset_index(drop=True), day[keep].reset_index(drop=True),
                  X[[c for c in features.CATS]][keep].reset_index(drop=True))
    del cong, dep, X, wx, at, ex; gc.collect()

A = pd.concat([res['training'][0], res['ranking'][0]], ignore_index=True)
ym = pd.concat([res['training'][1], res['ranking'][1]], ignore_index=True)
apt = pd.concat([res['training'][2], res['ranking'][2]], ignore_index=True)
day = pd.concat([res['training'][3], res['ranking'][3]], ignore_index=True)
summarize('base+cong+wx+atfm+extra', A, ym, apt, out)
# per-day missing for wx/atfm in 2026
for c in ['wx_tmpc', 'wx_sknt', 'atfm_dly_total', 'wx_age_min']:
    if c not in A: continue
    x = A[c].isna() if c != 'wx_age_min' else (A[c] > 60)
    g = x.groupby([apt, day]).mean()
    g = g[g > 0.05]
    print(f'\n{c}: airport-days with >5% missing (age>60 for wx_age_min):', len(g), file=out)
    print(g.round(3).to_string(), file=out)
# per-day mean of n_taxiing_out / dep_apt_to_60m in 2026 relative to airport-month median (feed glitches)
c = 'dep_apt_to_60m'
g = A[c].groupby([apt, ym, day]).mean().rename('v').reset_index()
g['med'] = g.groupby(['level_0' if 'level_0' in g else g.columns[0], g.columns[1]]).v.transform('median')
print('\ncongestion day means < 60% of airport-month median:', file=out)
print(g[g.v < 0.6 * g.med].to_string(), file=out)
# base categorical: PIER unseen
C = pd.concat([res['training'][4], res['ranking'][4]], ignore_index=True)
for c in ['PIER', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', 'AIRCRAFT_TYPE_mvt', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt']:
    trv = set(res['training'][4][c].unique())
    un = ~C[c].isin(trv)
    print(f'\nbase cat {c}: share of values unseen in 2025 (training months 1,2,6,7 vocabulary!) by apt/month', file=out)
    print(un.groupby([apt, ym]).mean().unstack().reindex(columns=MON).round(4).to_string(), file=out)
out.close()
