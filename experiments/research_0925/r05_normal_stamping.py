"""r05 -- are stamped-at-SCHED rows hiding inside the 'ordinary flights' pool (NM record, y <= 1800)?
Per airport, scored months (2025 OOF, v15 stand-in from r01): SSE share and RMSE of ordinary rows split by
stamped = |y - off_sched| <= 120 s and by departure delay (off_sched - y proxy is not observable; we use
off_sched bins, which ARE observable). Read-only.
    python experiments/research_0925/r05_normal_stamping.py --v15s <dir>/v15s_oof.npy
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
ap = argparse.ArgumentParser(); ap.add_argument('--v15s', required=True); a = ap.parse_args()
d = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'month', 'y', 'off_sched', 'off_aobt3'])
p = np.load(a.v15s).astype(float); y = d.y.to_numpy().astype(float); so = d.off_sched.to_numpy(); oa = d.off_aobt3.to_numpy()
b = np.load(CACHE / 'v11' / 'base.npz'); P, D = b['P'], b['D']
S = np.isin(d.month.to_numpy(), [1, 7]); apt = d.APT.astype(str).to_numpy()
unreach = D & ~P & (y > 7200)
e = (y - p) ** 2; tot = e[S & ~unreach].sum()
normal = ~D & (y <= 1800); st = np.abs(y - so) <= 120
print('scored months, ordinary rows (NM record, y<=1800): % of reachable scored SSE | RMSE | n')
rows = []
for ap_ in sorted(set(apt)):
    k = S & normal & (apt == ap_)
    r = lambda m: np.sqrt(e[m].mean()) if m.sum() else np.nan
    rows.append(dict(apt=ap_, n=int(k.sum()), stamped_share_rows=(k & st).sum() / k.sum(),
                     sse_pct_stamped=100 * e[k & st].sum() / tot, sse_pct_unstamped=100 * e[k & ~st].sum() / tot,
                     rmse_stamped=r(k & st), rmse_unstamped=r(k & ~st),
                     rmse_offsched_gt600=r(k & (so > 600)), rows_offsched_gt600=(k & (so > 600)).sum() / k.sum(),
                     mean_res_stamped=np.mean((y - p)[k & st]) if (k & st).sum() else np.nan))
print(pd.DataFrame(rows).round(3).to_string(index=False))
k = S & normal
print(f'ALL: stamped rows {100 * (k & st).sum() / k.sum():.1f}% of ordinary rows carry {100 * e[k & st].sum() / tot:.2f}% '
      f'of reachable scored SSE (unstamped {100 * e[k & ~st].sum() / tot:.2f}%)')
kk = S & normal & (apt == 'LIRF')
for lo, hi in ((-1e9, 0), (0, 600), (600, 1200), (1200, 1800), (1800, 1e9)):
    m = kk & (so > lo) & (so <= hi)
    print(f'  LIRF ordinary, off_sched ({lo:>6.0f},{hi:>6.0f}]: n {m.sum():6d}  stamped {100 * (m & st).sum() / max(m.sum(), 1):5.1f}%  '
          f'RMSE {np.sqrt(e[m].mean()):6.1f}  SSE {100 * e[m].sum() / tot:5.2f}%  mean res {np.mean((y - p)[m]):+7.1f}')
