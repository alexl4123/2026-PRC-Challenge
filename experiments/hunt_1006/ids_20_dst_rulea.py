"""ids_20 -- Rule A (convex): route-D departure linked to a destination NM arrival (same callsign/ADEP/ADES, arrival
0-24 h after take-off), NM flight not matched to any DEP row, ADEP != ADES, and MVT - LOBT_dst <= 3594
=> the 'inside window' set is y in [L, U] with L <= -12, so (if the one-to-one window join is the only failure cause)
   y >= U = MVT - LOBT_dst + 3606.  Exactness on 2025, OOF effect of clipping v28 from below, 2026 rows moved."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 280); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
m = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2025.parquet')
print('exceptions (inside window) with ADEP==ADES:', pd.crosstab(m.ADEP_mvt == m.ADES_mvt, m.inside).to_string())
print(m[m.inside][['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'used', 'twin', 'ft', 'mvt_m_lobt', 'y']].to_string())
# attach v28 OOF via MVT_ID
tr = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
tr['f'] = z['f']; tr['P'] = z['P']; tr['month'] = z['month']; tr['yz'] = z['y']
m = m.merge(tr, on='MVT_ID_mvt', how='left')
assert np.allclose(m.y, m.yz)
def rule(df):
    return (~df.used) & (df.ADEP_mvt != df.ADES_mvt) & (df.mvt_m_lobt <= 3594)
for name, sel in [('A: not used, ADEP!=ADES, MVT-LOBT<=3594', rule(m)),
                  ('A + aobt consistent', rule(m) & (m.mvt_m_aobt > -60) & (m.mvt_m_aobt < 7200)),
                  ('A without used filter', (m.ADEP_mvt != m.ADES_mvt) & (m.mvt_m_lobt <= 3594))]:
    r = m[sel].copy()
    r['U'] = r.mvt_m_lobt + 3606
    exc = r.y < r.U
    print(f'\n{name}: n={len(r)}  exceptions (y < U) = {exc.sum()}   P rows {r.P.sum()}  by apt', r.APT.value_counts().to_dict())
    if exc.sum():
        print(r[exc][['APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'y', 'U', 'used', 'twin', 'cs_eq']].to_string())
    g = np.maximum(r.f, r.U)
    mv = g - r.f
    d = (r.y - g) ** 2 - (r.y - r.f) ** 2
    jj = r.month.isin([1, 7])
    print(f'  OOF: rows moved {np.sum(mv > 0)}, sum move^2 {np.sum(mv ** 2):.3e}, dSSE 12m {d.sum():.3e} (rows worse {np.sum(d > 0)}), '
          f'dSSE Jan+Jul {d[jj].sum():.3e};  dRMSE 12m ~ {d.sum() / 2085047 / (2 * 234.37):+.3f} s')
    print(r[mv > 0][['APT', 'FLIGHT_mvt', 'month', 'P', 'y', 'f', 'U']].round(0).to_string())
# 2026
k = pd.read_parquet(H + 'cache/hunt/ids_19_dstlink_2026.parquet')
v = pd.read_parquet(H + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
k = k.merge(v, on='MVT_ID_mvt', how='left')
r = k[rule(k)].copy()
r['U'] = r.mvt_m_lobt + 3606
r['P'] = r.off_sched > 7200
mv = np.maximum(r.f, r.U) - r.f
print(f'\n2026 rule-A rows: {len(r)} (P {r.P.sum()}), rows v28 would move: {np.sum(mv > 0)}, sum move^2 {np.sum(mv ** 2):.3e} '
      f'-> guaranteed RMSE drop >= {235.3996 - np.sqrt(235.3996 ** 2 - np.sum(mv ** 2) / 344841):.3f} s if the rule holds')
print(r.assign(move=mv)[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'off_sched', 'mvt_m_lobt',
                          'mvt_m_aobt', 'ft', 'f', 'U', 'move', 'twin']].sort_values('move', ascending=False).head(40).round(0).to_string())
