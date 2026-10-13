"""Candidate files (scratch, cache/hunt/ only): v28 + A (one-sided exclusion) + B (two-sided, stamp excluded) + C (LEBL/LEMD cap),
and the same + E1 [statistical]. Re-derives the links from the raw caches; checks; 2025 OOF of the identical code path."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 200)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
W, CAP = 3606, 3606
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT', 'SCHED', 'LOBT', 'CALLSIGN_flt', 'FLIGHT_RULE_mvt']

def rules(yr, f):
    """Return new predictions (A+B+C and +E1) for the departure rows of one file, in file order."""
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=C); a = pd.read_parquet(H + f'{yr}_arr.parquet', columns=C)
    for x in (d, a):
        for c in ['APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt']: x[c] = x[c].astype(str)
    d['i'] = np.arange(len(d)); used = set(d.FLIGHT_ID_mvt.dropna().unique())
    D = d[d.LOBT.isna() & (d.FLIGHT_RULE_mvt.astype(str) == 'I') & (d.APT != d.ADES_mvt)]
    A = a[a.LOBT.notna() & ~a.FLIGHT_ID_mvt.isin(used)][['FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt', 'APT', 'MVT', 'LOBT']]
    A = A.rename(columns={c: 'a_' + c for c in A.columns})
    L = []
    for key in ('a_FLIGHT_mvt', 'a_CALLSIGN_flt'):
        m = D.merge(A, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=[key, 'a_ADEP_mvt', 'a_APT'])
        m['t'] = m.a_MVT - m.MVT; L.append(m[(m.t > 0) & (m.t < 86400)])
    m = pd.concat(L).sort_values('t').drop_duplicates('i').reset_index(drop=True)
    off = (m.MVT - m.a_LOBT).to_numpy(); lo, hi = off - W, off + W; osd = (m.MVT - m.SCHED).to_numpy(); i = m.i.to_numpy()
    one = lo < -12; stamp_excl = (osd >= lo) & (osd <= hi)
    g = f.astype(float).copy()
    g[i[one]] = np.maximum(f[i[one]], np.ceil(hi[one]))                                   # A
    b = ~one & stamp_excl; g[i[b]] = np.minimum(f[i[b]], np.floor(np.maximum(lo[b], 0)))  # B
    sp = d.APT.isin(['LEBL', 'LEMD']).to_numpy(); g[sp] = np.minimum(g[sp], CAP)            # C
    e = g.copy(); s = one & (osd > hi) & (m.APT == 'LIRF').to_numpy(); e[i[s]] = np.maximum(g[i[s]], osd[s])  # E1 [E]
    info = pd.DataFrame({'i': i, 'APT': m.APT, 'FLIGHT_mvt': m.FLIGHT_mvt, 'one': one, 'B': b, 'E1': s, 'lo': lo, 'hi': hi, 'off_sched': osd})
    return g, e, info, d

# ---- 2025 OOF (identical code path) ----
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
g, e, info, d = rules('tr', f)
r = lambda h, k: np.sqrt(np.mean((y[k] - h[k]) ** 2)); ALL = np.ones(len(y), bool); held = ~np.isin(month, [3, 6, 9, 12]); S = np.isin(month, [1, 7])
for nm, h in (('A+B+C', g), ('A+B+C+E1 [E]', e)):
    dd = (y - h) ** 2 - (y - f) ** 2; ch = h != f
    top = np.sort(-dd[ch])[:10].sum() / -dd[ch].sum() if ch.any() else 0
    print(f'2025 OOF {nm}: rows changed {ch.sum()}, worse {(dd > 0).sum()}, 12m {r(h, ALL) - r(f, ALL):+.4f}, held-out {r(h, held) - r(f, held):+.4f}, Jan+Jul {r(h, S) - r(f, S):+.4f}, '
          f'Jan {r(h, month == 1) - r(f, month == 1):+.4f}, Jul {r(h, month == 7) - r(f, month == 7):+.4f}, months better {sum(r(h, month == k) < r(f, month == k) for k in range(1, 13))}/12, top-10 rows share of gain {top:.2f}')
# exactness of the premises on 2025
yy = y[info.i]; print('2025 premises: A rows', info.one.sum(), 'with y < hi:', int((yy[info.one.to_numpy()] < info.hi[info.one]).sum()),
                     '| B rows', info.B.sum(), 'with y > lo:', int((yy[info.B.to_numpy()] > info.lo[info.B]).sum()),
                     '| LEBL/LEMD rows with y > 3606:', int((y[d.APT.isin(['LEBL', 'LEMD']).to_numpy()] > CAP).sum()))
# ---- 2026 ----
rk = pd.read_parquet(H + 'ra_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'MVT', 'LOBT'])
v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
assert np.isfinite(v).all()
g6, e6, info6, d6 = rules('ra', v)
# consistency with the LOBT window for capped rows
ol = (rk.MVT - rk.LOBT).to_numpy(); wl = np.ceil(np.maximum(0, ol - W)); capped = d6.APT.isin(['LEBL', 'LEMD']).to_numpy() & (v > CAP)
assert not (capped & (wl > CAP)).any(), 'cap conflicts with the LOBT window'
assert (g6[np.isfinite(ol)] >= wl[np.isfinite(ol)]).all()
for nm, h in (('A+B+C', g6), ('A+B+C+E1 [E]', e6)):
    ch = h != v; s = h - v
    print(f'2026 {nm}: rows moved {ch.sum()} (route D {(ch & rk.LOBT.isna().to_numpy()).sum()}, LEBL/LEMD cap {(ch & capped).sum()}), sum sq {np.sum(s ** 2):.4g}, '
          f'RMSE bound if rules hold {np.sqrt(235.3996 ** 2 - np.sum(s ** 2) / len(v)):.4f} ({np.sqrt(235.3996 ** 2 - np.sum(s ** 2) / len(v)) - 235.3996:+.4f})')
    mm = pd.DataFrame({'APT': d6.APT, 'mon': pd.to_datetime(d6.MVT, unit='s').dt.month, 's': s}).groupby(['APT', 'mon']).s.mean()
    print('   airport x month mean shift range', round(mm.min(), 2), round(mm.max(), 2))
    out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.to_numpy(), 'TAXITIME_SEC_mvt': h.astype(np.int64)})
    out.to_parquet(H + ('cand_v28_ABC.parquet' if nm == 'A+B+C' else 'cand_v28_ABC_E1.parquet'), index=False)
ch = (g6 != v) | (e6 != v)
x = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt[ch].to_numpy(), 'APT': d6.APT[ch].to_numpy(), 'FLIGHT_mvt': d6.FLIGHT_mvt[ch].to_numpy(), 'v28': v[ch], 'ABC': g6[ch], 'E1': e6[ch]})
x = x.merge(info6.assign(MVT_ID_mvt=rk.MVT_ID_mvt.to_numpy()[info6.i])[['MVT_ID_mvt', 'one', 'B', 'lo', 'hi', 'off_sched']], on='MVT_ID_mvt', how='left')
print(x.sort_values(['one', 'APT']).to_string())
x.to_csv(H + 'cand_moves.csv', index=False)
