"""m04 -- how big is our 2026 BULK error? Estimated from submissions we already made.

DIAGNOSTIC ONLY. No new submission; it uses 2026 labels only through our own public scores
(the live scores of v7/v8/v9
already on the board) and asks what they imply about where our 2026 SSE lives.

For two submissions a, b differing by delta = b - a on rows K:
    SSE_b = SSE_a - 2 sum_K r_a*delta + sum_K delta^2        (r_a = y - a)
so the live scores give  sum_K r_a*delta  exactly (to score rounding).
v7 -> v8 and v8 -> v9 change ONLY rows outside P (the network blend).  If the
cosine between r_a and delta on the clean bulk is the same in 2026 as in the 2025
OOF (a stability assumption, stated), then
    SSE_bulk_2026 ~ (sum r*delta)^2 / (cos^2 * sum delta^2).
A second estimator assumes the RELATIVE bulk-SSE gain transfers:
    SSE_bulk_2026 ~ dSSE_live * SSE_bulk_oof / dSSE_oof.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS

M = CACHE / 'major'
N = 344841
LIVE = {'v5': 321.2771, 'v6': 308.1903, 'v7': 304.351, 'v8': 297.829, 'v9': 297.3414}
SSE = {k: v * v * N for k, v in LIVE.items()}

rk = pd.read_parquet(M / 'dep_ranking.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'off_aobt3', 'off_sched'])
sub = {}
for v in LIVE:
    s = pd.read_parquet(OUTPUTS / f'understanding-accordion_{v}.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    sub[v] = s.reindex(rk.MVT_ID_mvt).to_numpy().astype(float)
    assert np.isfinite(sub[v]).all()
Dr = rk.off_aobt3.isna().to_numpy(); Pr = Dr & (rk.off_sched.to_numpy() > 7200)
print('ranking P rows', Pr.sum(), ' route D', Dr.sum())

# ---- OOF analogue ------------------------------------------------------------------
d = pd.read_parquet(M / 'dep_training.parquet', columns=['y', 'month', 'off_aobt3', 'off_sched'])
y = d.y.to_numpy(); mon = d.month.to_numpy()
D = d.off_aobt3.isna().to_numpy(); P = D & (d.off_sched.to_numpy() > 7200)
o = np.load(CACHE / 'exp62_oof.npy'); net = np.load(CACHE / 'blend_oof_mix_nn3.npy')[1].astype(float)
v7o = o[2]
v8o = np.where(P, v7o, 0.5 * v7o + 0.5 * net)
w = np.where(D, 0.5, 0.65)
v9o = np.where(P, v7o, (1 - w) * v7o + w * net)
oof = {'v7': v7o, 'v8': v8o, 'v9': v9o}

for a, b in [('v7', 'v8'), ('v8', 'v9'), ('v7', 'v9')]:
    dl = sub[b] - sub[a]
    print(f'  rows changed: inside P {(np.abs(dl[Pr])>0.5).sum()} (sum delta^2 {(dl[Pr]**2).sum():.3e}), route D notP {(np.abs(dl[Dr&~Pr])>0.5).sum()}, routes ABC {(np.abs(dl[~Dr])>0.5).sum()} (sum delta^2 {(dl[~Dr]**2).sum():.3e})')
    sdd = (dl ** 2).sum()
    srd = (SSE[a] - SSE[b] + sdd) / 2
    print(f'\n=== {a} -> {b}: live dSSE {SSE[b]-SSE[a]:.4e}  sum delta^2 {sdd:.4e}  sum r*delta {srd:.4e}')
    for scope, m in [('12m', np.ones(len(y), bool)), ('scored', np.isin(mon, [1, 7]))]:
        dlo = oof[b] - oof[a]
        if b == 'v8' or a == 'v7':
            pass
        ra = y - oof[a]
        for bname, kb in [('clean bulk (notP, y<=7200)', m & ~P & (y <= 7200)), ('all notP', m & ~P)]:
            cos = (ra[kb] * dlo[kb]).sum() / np.sqrt((ra[kb] ** 2).sum() * (dlo[kb] ** 2).sum())
            sse_bulk_oof = (ra[kb] ** 2).sum()
            dsse_oof = ((y[kb] - oof[b][kb]) ** 2).sum() - sse_bulk_oof
            est1 = srd ** 2 / (cos ** 2 * sdd)
            est2 = (SSE[b] - SSE[a]) * sse_bulk_oof / dsse_oof
            rows_live = (~Pr).sum()
            print(f'  [{scope:6s} {bname:27s}] cos {cos:.4f}  OOF bulk RMSE {np.sqrt(sse_bulk_oof/kb.sum()):6.1f}'
                  f'  rel gain {100*dsse_oof/sse_bulk_oof:+.3f}%  ->  est1 SSE {est1:.3e} (RMSE over notP rows '
                  f'{np.sqrt(est1/rows_live):6.1f})   est2 SSE {est2:.3e} ({np.sqrt(est2/rows_live):6.1f})')

print('\nlive total SSE v9', f'{SSE["v9"]:.4e}', ' leader 234.51 ->', f'{234.51**2*N:.4e}')
