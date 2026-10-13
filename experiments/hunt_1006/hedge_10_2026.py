"""2026 (label-free): zone-P hedge rows of v28 and what rules A1 / B / C would move.
Rebuilds p, pb, corr of the zone-P rows exactly as assemble.py does and checks them against the shipped v28 file."""
import sys
sys.path.insert(0, "/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026")
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import peak_rss_gb
import assemble as A
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
C = A.Components()
t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
pb = 0.5 * C.fit('w2_final', 'mix')[rows] + 0.5 * C.member('seq')[rows]
so = C.off_sched[rows]; corr = C.corr[rows]
v = A.shipped(C, 28)
rebuilt = np.rint(np.clip(p * so + (1 - p) * pb + corr, 0, A.HI))
print('zone P rows', len(rows), ' rebuilt == v28:', int((rebuilt == v[rows]).sum()), ' max |diff|', np.abs(rebuilt - v[rows]).max())
apt = C.apt[rows]; mon = C.month[rows]
hedge = p * (1 - p) * (so - pb) ** 2
tot = ((v - 0) ** 0).sum()  # placeholder
print('\n2026 zone P by airport: rows, p mean, rows with 0.05<p<0.95, sum p(1-p)(Hs-Hn)^2 (expected hedge SSE if p calibrated)')
df = pd.DataFrame(dict(apt=apt, month=mon, p=p, so=so, pb=pb, hedge=hedge))
print(df.groupby(['apt']).agg(n=('p', 'size'), p_mean=('p', 'mean'), mid=('p', lambda x: int(((x > .05) & (x < .95)).sum())),
                              hedge=('hedge', 'sum')).to_string())
print('total expected hedge SSE (zone P) %.3e = %.1f%% of a 235.4^2*N file SSE' % (hedge.sum(), 100 * hedge.sum() / (235.4 ** 2 * len(v))))
print('LIRF Jan/Jul rows:', np.bincount(mon[apt == 'LIRF'], minlength=8)[[1, 7]].tolist())
# --- rule A1 (dest-NM anti-window): stamp excluded -> p = 0
L = pd.read_parquet(CACHE / 'hunt' / 'hedge_plink26.parquet')
L = L[L.P & (L.FLIGHT_RULE_mvt.astype(str) == 'I')]
k = pd.Series(np.arange(len(rows)), index=rows); ix = k.reindex(L.pos).to_numpy()
olo = np.full(len(rows), np.nan); olo[ix] = (L.mvt - L.lobt).to_numpy()
s_in = (so >= olo - 3606) & (so <= olo + 3606)
pA1 = np.where(s_in, 0.0, p)
qA1 = np.rint(np.clip(pA1 * so + (1 - pA1) * pb + corr, 0, A.HI))
mv = qA1 != v[rows]
print(f'\nA1: linked IFR zone-P rows {len(L)} (LIRF {int((L.adep == "LIRF").sum())}); stamp excluded {int(s_in.sum())}; rows moved {int(mv.sum())}; '
      f'|move| sum^2 {((qA1 - v[rows]) ** 2)[mv].sum():.3e}; max |move| {np.abs(qA1 - v[rows]).max():.0f}')
print(pd.DataFrame(dict(apt=apt, month=mon, p=p.round(3), off_sched=so, pb=pb.round(), v28=v[rows], new=qA1))[mv].sort_values('p').tail(15).to_string())
# --- rule B / C
tw = pd.read_parquet(CACHE / 'hunt' / 'hedge_lirf_tw26.parquet')
tw = tw[tw.P]
ix2 = k.reindex(tw.pos).to_numpy(); ddu = np.zeros(len(rows), int); ddu[ix2] = tw.dd_utc.to_numpy()
ddl = np.zeros(len(rows), int); ddl[ix2] = tw.dd_loc.to_numpy()
B = (apt == 'LIRF') & (ddu >= 1)
qB = np.rint(np.clip(np.where(B, 1.0, p) * so + (1 - np.where(B, 1.0, p)) * pb + corr, 0, A.HI))
mvB = qB != v[rows]
print(f'\nB: LIRF zone-P rows with UTC date crossing {int(B.sum())}; moved {int(mvB.sum())}; sum move^2 {((qB - v[rows]) ** 2).sum():.3e}')
print(pd.DataFrame(dict(month=mon, p=p.round(3), dd_loc=ddl, off_sched=so, pb=pb.round(), v28=v[rows], B=qB))[B].sort_values('off_sched').to_string())
band = B & (so > 50000) & (so <= 95000)
for q in (0.5, 0.67):
    qC = qB.copy()
    qC[band] = np.rint(q * (86400 * np.maximum(ddl[band], 1) + pb[band]) + (1 - q) * so[band] + corr[band])
    d = qC - v[rows]
    print(f'C q={q}: band rows {int(band.sum())}: moves {d[band].round().tolist()}  (months {mon[band].tolist()})')
    # expected-value arithmetic per row under the 2025 class shares in the band (10 day-shift : 5 stamp)
    Hs = so[band]; Hd = 86400 * np.maximum(ddl[band], 1) + pb[band]; cur = v[rows][band] - corr[band]; new = qC[band] - corr[band]
    for share in (0.0, 10 / 15, 1.0):
        e_cur = share * (Hd - cur) ** 2 + (1 - share) * (Hs - cur) ** 2
        e_new = share * (Hd - new) ** 2 + (1 - share) * (Hs - new) ** 2
        tot0 = 235.3996 ** 2 * 344841
        print(f'   if day-shift share = {share:.2f}: dSSE {(e_new - e_cur).sum():+.3e} -> dRMSE {np.sqrt((tot0 + (e_new - e_cur).sum()) / 344841) - 235.3996:+.2f} s')
print('peak GB', peak_rss_gb())
