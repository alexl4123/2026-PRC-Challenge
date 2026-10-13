"""ids_08 -- duplicate departure rows ('twins': same airport, callsign, SCHED). Label structure on 2025, v28 OOF
errors on them, discrimination of first/second by ID order vs by NM-record presence, and 2026 counts + v28 preds."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
     'LOBT_flt', 'STAND_mvt']
def prep(which):
    d = load(which, phase='DEP', columns=C)
    d['APT'] = d.APT.astype(str); d['FLIGHT_mvt'] = d.FLIGHT_mvt.astype(str)
    for c in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'LOBT_flt']:
        v = d[c].values.astype('datetime64[s]').astype('float64'); v[d[c].isna().values] = np.nan; d[c[:5]] = v
    d['y'] = d.MVT_T - d.BLOCK
    d['off_sched'] = d.MVT_T - d.SCHED
    d['nm'] = d.LOBT_.notna()
    return d
tr = prep('training')
mt = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
mt['f'] = z['f']; mt['P'] = z['P']; mt['D'] = z['D']; mt['yz'] = z['y']; mt['month'] = z['month']
tr = tr.merge(mt, on='MVT_ID_mvt', how='left')
assert np.allclose(tr.y, tr.yz)
rk = prep('ranking')
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
rk = rk.merge(v, on='MVT_ID_mvt', how='left')
rk['P'] = (~rk.nm) & (rk.off_sched > 7200)

def twins(d, label):
    g = d.groupby(['APT', 'FLIGHT_mvt', 'SCHED']).MVT_ID_mvt.transform('size')
    t = d[g >= 2].copy()
    t['gsize'] = g[g >= 2]
    t = t.sort_values(['APT', 'FLIGHT_mvt', 'SCHED', 'MVT_ID_mvt'])
    t['gid'] = t.groupby(['APT', 'FLIGHT_mvt', 'SCHED']).ngroup()
    t['idrank'] = t.groupby('gid').cumcount()
    t['dmvt'] = t.groupby('gid').MVT_T.transform(lambda s: s.max() - s.min())
    t['n_nm'] = t.groupby('gid').nm.transform('sum')
    print(f'\n===== {label}: twin groups {t.gid.nunique()} rows {len(t)}; size dist', t.groupby('gid').size().value_counts().to_dict())
    print('  groups by max-min MVT spread: <=120s', (t.groupby('gid').dmvt.first() <= 120).sum(),
          ' <=1h', (t.groupby('gid').dmvt.first() <= 3600).sum())
    print('  by airport (groups):', t.groupby('gid').APT.first().value_counts().to_dict())
    print('  n_nm per group:', t.groupby('gid').n_nm.first().value_counts().to_dict())
    return t
tt = twins(tr, '2025 training')
t2 = tt[(tt.gsize == 2) & (tt.dmvt <= 600)].copy()
first = t2[t2.idrank == 0].set_index('gid'); second = t2[t2.idrank == 1].set_index('gid')
print('\n2025 pairs with MVT spread <= 600 s:', len(first))
print('  BLOCK order agrees with ID order:', np.mean(first.BLOCK < second.BLOCK), ' ties', np.mean(first.BLOCK == second.BLOCK))
print('  first row nm / second row nm crosstab:\n', pd.crosstab(first.nm, second.nm).to_string())
print('  first y quantiles', np.quantile(first.y, [.1, .25, .5, .75, .9]).round(0))
print('  second y quantiles', np.quantile(second.y, [.1, .25, .5, .75, .9]).round(0))
print('  first: share |BLOCK-SCHED|<=60', np.mean(np.abs(first.BLOCK - first.SCHED) <= 60))
# does nm presence identify the 'late block' row?
both = pd.DataFrame({'y1': first.y, 'y2': second.y, 'nm1': first.nm, 'nm2': second.nm, 'f1': first.f, 'f2': second.f,
                     'P1': first.P, 'P2': second.P, 'D1': first.D, 'D2': second.D, 'apt': first.APT, 'm': first.month,
                     'os1': first.off_sched, 'os2': second.off_sched, 'mvt1': first.MVT_T, 'mvt2': second.MVT_T})
both['long_is_first'] = both.y1 > both.y2
print('  long row is the first by ID:', both.long_is_first.mean())
x = both[both.nm1 != both.nm2]
print('  pairs where exactly one row has NM:', len(x), ' long row is the non-NM row:', np.mean(np.where(x.nm1, x.y2 > x.y1, x.y1 > x.y2)))
print('  MVT order: second MVT earlier share', np.mean(both.mvt2 < both.mvt1))
# v28 OOF errors on twin rows vs oracle
sse_now = ((both.y1 - both.f1) ** 2 + (both.y2 - both.f2) ** 2)
print('  v28 OOF SSE on these pairs', sse_now.sum(), ' per pair mean', sse_now.mean().round(0))
print('  rows in P:', both.P1.sum() + both.P2.sum(), ' rows in D:', both.D1.sum() + both.D2.sum())
print(both.assign(e1=both.y1 - both.f1, e2=both.y2 - both.f2)[['apt', 'm', 'y1', 'f1', 'e1', 'y2', 'f2', 'e2', 'nm1', 'nm2', 'P1', 'P2', 'os1', 'os2']].round(0).head(60).to_string())
both.to_parquet(H + 'cache/hunt/ids_08_pairs2025.parquet')
tt.to_parquet(H + 'cache/hunt/ids_08_twins2025.parquet')

rt = twins(rk, '2026 ranking')
r2 = rt[(rt.gsize == 2) & (rt.dmvt <= 600)]
f_ = r2[r2.idrank == 0].set_index('gid'); s_ = r2[r2.idrank == 1].set_index('gid')
rb = pd.DataFrame({'apt': f_.APT, 'fl': f_.FLIGHT_mvt, 'f1': f_.f, 'f2': s_.f, 'nm1': f_.nm, 'nm2': s_.nm, 'P1': f_.P, 'P2': s_.P,
                   'os1': f_.off_sched, 'os2': s_.off_sched, 'sched': pd.to_datetime(f_.SCHED, unit='s'),
                   'lo1': f_.MVT_T - f_.LOBT_, 'lo2': s_.MVT_T - s_.LOBT_, 'id1': f_.MVT_ID_mvt, 'id2': s_.MVT_ID_mvt})
print('\n2026 pairs (MVT spread <= 600):', len(rb))
print(rb.round(0).to_string())
rt.to_parquet(H + 'cache/hunt/ids_08_twins2026.parquet')
