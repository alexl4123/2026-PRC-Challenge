"""Doubled-prefix flight numbers (ITYTY674, WMTMT2UK, RYRR90BN) and BLOCK = SCHED stamping inside zone P (route D, off_sched > 7200)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
def dup(s):
    s = str(s)
    if len(s) < 5 or not s[:4].isalpha(): return False
    return s[2] == s[3] or s[3:5] == s[1:3] or s[3:6] == s[0:3]
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'FLIGHT_mvt', 'MVT', 'BLOCK', 'SCHED', 'LOBT']); d['APT'] = d.APT.astype(str)
d['y'] = d.MVT - d.BLOCK; d['os'] = d.MVT - d.SCHED; d['dup'] = d.FLIGHT_mvt.map(dup)
P = d[d.LOBT.isna() & (d.os > 7200)].copy(); P['stamped'] = (P.y - P.os).abs() <= 6; P['tail'] = P.y > 7200
print(P.groupby(['APT', 'dup']).agg(n=('y', 'size'), stamped=('stamped', 'mean'), tail=('tail', 'mean')).round(3).to_string())
print('all deps dup share by APT:', d.groupby('APT').dup.mean().round(4).to_dict())
r = pd.read_parquet(H + 'ra_dep.parquet', columns=['APT', 'FLIGHT_mvt', 'MVT', 'SCHED', 'LOBT']); r['APT'] = r.APT.astype(str); r['dup'] = r.FLIGHT_mvt.map(dup)
rP = r[r.LOBT.isna() & ((r.MVT - r.SCHED) > 7200)]
print('2026 zone P: dup rows by APT', rP.groupby('APT').dup.agg(['size', 'sum']).to_dict())
