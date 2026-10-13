"""Zone P / route D: link route-D departures to their own arrival row at another of the 10 airports that carries an NM
record (LOBT, AOBT_3, IOBT, EOBT_1 of the SAME flight).  Normalises LIRF's doubled flight strings ('ITYTY672' -> 'ITY672')
and matches the departure string against the arrival's FLIGHT_mvt OR its NM CALLSIGN.
Then: (1) does the departure's BLOCK lie outside [LOBT_a - 3606, LOBT_a + 3606] (2025, exact?), (2) for zone P rows,
which hypothesis (stamp = off_sched, normal = bulk, day-shift) does that exclusion kill, (3) 2026 counts."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load, peak_rss_gb
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
EP = pd.Timestamp('1970-01-01', tz='UTC')


def norm(s):
    s = '' if s is None or s != s else str(s)
    if len(s) >= 5 and s[:3].isalpha() and s[3].isalpha():
        for k in (3, 2, 1):
            if s[3:3 + k] == s[3 - k:3]:
                return s[:3] + s[3 + k:]
    return s


def sec(x):
    return (x - EP).dt.total_seconds().to_numpy()


def build(which):
    dc = ['MVT_ID_mvt', 'FLIGHT_mvt', 'APT', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'FLIGHT_RULE_mvt']
    if which == 'training':
        dc.append('BLOCK_TIME_UTC_mvt')
    d = load(which, phase='DEP', columns=dc)
    d['pos'] = np.arange(len(d))
    d = d[d.LOBT_flt.isna()].copy()
    d['fl'] = [norm(x) for x in d.FLIGHT_mvt.astype(object)]
    d['ades'] = d.ADES_mvt.astype(str); d['adep'] = d.APT.astype(str)
    d['mvt'] = sec(d.MVT_TIME_UTC_mvt); d['sch'] = sec(d.SCHED_TIME_UTC_mvt)
    d['blk'] = sec(d.BLOCK_TIME_UTC_mvt) if which == 'training' else np.nan
    d = d[['MVT_ID_mvt', 'pos', 'fl', 'adep', 'ades', 'mvt', 'sch', 'blk', 'FLIGHT_RULE_mvt']]
    ac = ['FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'IOBT_flt', 'EOBT_1_flt']
    a = load(which, phase='ARR', columns=ac)
    a = a[a.LOBT_flt.notna() & a.ADEP_mvt.astype(str).isin(APTS)].copy()
    out = pd.DataFrame({'adep': a.ADEP_mvt.astype(str).to_numpy(), 'ades': a.APT.astype(str).to_numpy(),
                        'amvt': sec(a.MVT_TIME_UTC_mvt), 'lobt': sec(a.LOBT_flt), 'aobt': sec(a.AOBT_3_flt),
                        'iobt': sec(a.IOBT_flt), 'eobt': sec(a.EOBT_1_flt)})
    k1 = out.assign(fl=[norm(x) for x in a.FLIGHT_mvt.astype(object)])
    k2 = out.assign(fl=[norm(x) for x in a.CALLSIGN_flt.astype(object)])
    A = pd.concat([k1, k2]).drop_duplicates()
    del a
    m = d.merge(A, on=['fl', 'adep', 'ades'])
    t = m.amvt - m.mvt
    m = m[(t > 0) & (t < 86400)].assign(t=t[(t > 0) & (t < 86400)]).sort_values('t').drop_duplicates('pos')
    return d, m


V = np.load(CACHE / 'w2' / 'v28s_oof.npz', allow_pickle=True)
y, P, D, f = V['y'], V['P'], V['D'], V['f']
W = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
p = np.full(len(y), np.nan); p[W['Pidx']] = W['p14']; pb = np.full(len(y), np.nan); pb[W['Pidx']] = W['pbP']
d, m = build('training')
m['y'] = m.mvt - m.blk; m['f'] = f[m.pos]; m['P'] = P[m.pos]; m['p'] = p[m.pos]; m['pb'] = pb[m.pos]
m['b_l'] = m.blk - m.lobt; m['excl'] = m.b_l.abs() > 3606
m['off_sched'] = m.mvt - m.sch; m['off_lobt'] = m.mvt - m.lobt; m['off_aobt'] = m.mvt - m.aobt
m['cls'] = np.where(np.abs(m.y - m.off_sched) <= 120, 'stamp', np.where(m.y > 43200, 'dayshift', np.where(m.y <= 7200, 'normal', 'other')))
print(f'2025 route-D deps {len(d):,}; intra-network {d.ades.isin(APTS).sum():,}; linked to an NM arrival {len(m):,} '
      f'(zone P {int(m.P.sum())}, of which LIRF {int((m.P & (m.adep == "LIRF")).sum())})')
print(' exclusion |BLOCK - LOBT_arr| > 3606 holds:', round(float(m.excl.mean()), 4), ' violations', int((~m.excl).sum()))
print(' by flight rule:'); print(m.groupby('FLIGHT_RULE_mvt', observed=True).excl.agg(['size', 'mean']).to_string())
print(' zone P linked rows: class x exclusion'); print(pd.crosstab(m[m.P].cls, m[m.P].excl))
# for zone P linked rows: where is each hypothesis relative to the window?  window in y-space: [off_lobt - 3606, off_lobt + 3606]
q = m[m.P].copy()
q['stamp_in_win'] = (q.off_sched >= q.off_lobt - 3606) & (q.off_sched <= q.off_lobt + 3606)
q['norm_in_win'] = (q.pb >= q.off_lobt - 3606) & (q.pb <= q.off_lobt + 3606)
q['aobt_in_win'] = (q.off_aobt >= q.off_lobt - 3606) & (q.off_aobt <= q.off_lobt + 3606)
print(' zone P linked: stamp hypothesis inside window (=> excluded) x class'); print(pd.crosstab([q.stamp_in_win, q.norm_in_win], q.cls))
print(q[['adep', 'ades', 'fl', 'cls', 'y', 'off_sched', 'off_lobt', 'off_aobt', 'p', 'pb', 'f', 'excl', 'FLIGHT_RULE_mvt']].sort_values('cls').to_string())
# AOBT_3 of the arrival record as the normal-taxi estimate: |y - off_aobt| for normal rows
nm = m[(m.cls == 'normal')]
print(' normal-class linked rows: |y - (MVT - AOBT_3_arr)| quantiles', np.quantile(np.abs(nm.y - nm.off_aobt), [.5, .9, .99]).round().tolist(),
      ' vs |y - f|', np.quantile(np.abs(nm.y - nm.f), [.5, .9, .99]).round().tolist())
m.to_parquet(CACHE / 'hunt' / 'hedge_plink25.parquet')
del d
d6, m6 = build('ranking')
rp = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
inP = rp['in_P'].astype(bool)
assert np.array_equal(rp['ids'], load('ranking', phase='DEP', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy())
m6['P'] = inP[m6.pos]
print(f'\n2026 route-D deps {len(d6):,}; zone P {int(inP.sum())}; intra-network {d6.ades.isin(APTS).sum():,}; linked {len(m6):,} '
      f'(zone P {int(m6.P.sum())}, LIRF P {int((m6.P & (m6.adep == "LIRF")).sum())})')
print(' 2026 zone P by airport:', pd.Series(d6.adep.to_numpy()[inP[d6.pos]]).value_counts().to_dict())
m6.to_parquet(CACHE / 'hunt' / 'hedge_plink26.parquet')
print('peak GB', peak_rss_gb())
