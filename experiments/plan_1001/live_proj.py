"""Diagnostic (read-only): what each past live score change implies about the 2026 residual along that version's shift.
SSE(old + c*s) = SSE_old - 2c*sum(s*r) + c^2*sum(s^2);  beta = sum(s*r)/sum(s^2) = (sum s^2 - dSSE)/(2 sum s^2).
beta > 0.5 <=> the shift helped; beta = 1 <=> the shift was the right size on average. NOT used to tune anything.
"""
import numpy as np, pandas as pd
O = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/outputs/understanding-accordion_v{}.parquet'
live = {16: 274.6519, 17: 251.1243, 18: 250.3448, 19: 248.7469, 20: 245.7300, 21: 245.4673, 22: 245.1468, 23: 244.6576,
        24: 244.2213, 25: 245.2878}
V = {}
for v in live:
    d = pd.read_parquet(O.format(v)).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].astype(float)
    V[v] = d
idx = V[24].index
N = len(idx)
print('N', N)
pairs = [(16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (19, 24), (17, 24)]
for a, b in pairs:
    s = (V[b].reindex(idx) - V[a].reindex(idx)).to_numpy()
    ss = (s ** 2).sum(); dsse = (live[b] ** 2 - live[a] ** 2) * N
    beta = (ss - dsse) / (2 * ss) if ss > 0 else np.nan
    nz = np.abs(s) > 0.5
    print(f'v{a}->v{b}: rows changed {nz.sum():7d}  rms shift {np.sqrt((s**2).mean()):7.2f}  sum s^2 {ss:.3e}  dSSE live {dsse:+.3e}  '
          f'beta {beta:+.2f}  (live {live[b]-live[a]:+.3f})')
