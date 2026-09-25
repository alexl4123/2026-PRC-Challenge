"""ESTIMATE of the scored-month RMSE gain from adsb.lol features (not a measurement).

    python experiments/adsb/adsb_scored_extrapolation.py MATCH_TRAIN.parquet COV_2026a.csv [COV_2026b.csv ...]
           [--lb 275.31]

Inputs: the training match file + its .gain_preds.parquet (adsb_gain_estimate.py, leave-one-day-out), and
label-free coverage tables of sample days in the scored months (adsb_match_measure.py --which ranking).

Per airport a, from the 2025 sample days:
  g_best[a] = mean over departures WITH an ADS-B off-block (t_best) of (e_base^2 - e_G^2)       [s^2]
  g_gnd[a]  = same over departures with ground points but no t_best
  airports with < 40 t_best rows borrow the pooled g_best (all airports' t_best rows).
For the scored months: share_best[a], share_gnd_only[a] = mean over the 2026 sample days (coverage CSVs),
n[a] = 2026 departures per airport from the ranking file. Then
  dMSE = sum_a n[a] * (share_best[a] * g_best[a] + share_gnd_only[a] * g_gnd[a]) / sum_a n[a]
  dRMSE ~ sqrt(LB^2 - dMSE) - LB     with LB = our current public score (275.31).
Two variants: 'same coverage as the 2025 sample' and '2026 coverage'. Both are estimates.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('match', type=Path)
    ap.add_argument('cov', nargs='+', type=Path)
    ap.add_argument('--lb', type=float, default=275.31)
    a = ap.parse_args()
    m = pd.read_parquet(a.match, columns=['MVT_ID_mvt', 'APT', 'ev_t_best', 'ev_n_seg', 'match'])
    g = pd.read_parquet(a.match.with_suffix('.gain_preds.parquet'))
    m = m.merge(g, on='MVT_ID_mvt')
    m['APT'] = m.APT.astype(str)
    m['dse'] = (m.y - m.base) ** 2 - (m.y - m.predG) ** 2
    m['best'] = m.ev_t_best.notna()
    m['gnd_only'] = (m.ev_n_seg.fillna(0) > 0) & ~m.best
    pooled = m.loc[m.best, 'dse'].mean()
    rows = []
    for apt, q in m.groupby('APT'):
        nb = int(q.best.sum())
        rows.append(dict(APT=apt, n25=len(q), share_best25=q.best.mean(), share_gnd25=q.gnd_only.mean(),
                         n_best25=nb, g_best=q.loc[q.best, 'dse'].mean() if nb >= 40 else pooled,
                         g_best_src='own' if nb >= 40 else 'pooled',
                         g_gnd=q.loc[q.gnd_only, 'dse'].mean() if q.gnd_only.sum() >= 40 else 0.0))
    r = pd.DataFrame(rows).set_index('APT')
    cov = pd.concat([pd.read_csv(p, index_col=0) for p in a.cov]).drop(index='ALL', errors='ignore')
    cov = cov.groupby(level=0)[['best_pct', 'ground_pct']].mean() / 100
    r['share_best26'] = cov.best_pct
    r['share_gnd26'] = (cov.ground_pct - cov.best_pct).clip(lower=0)
    rk = pq.read_table(REPO_ROOT / 'cache' / 'ranking.parquet', columns=['APT', 'PHASE_mvt'],
                       filters=[('PHASE_mvt', '=', 'DEP')]).to_pandas()
    r['n26'] = rk.APT.astype(str).value_counts()
    r = r.fillna({'share_best26': 0, 'share_gnd26': 0})
    N = r.n26.sum()
    out = {}
    for name, sb, sg in (('2025 sample coverage', 'share_best25', 'share_gnd25'),
                         ('2026 sample coverage', 'share_best26', 'share_gnd26')):
        dmse = float((r.n26 * (r[sb] * r.g_best + r[sg] * r.g_gnd)).sum() / N)
        out[name] = (dmse, np.sqrt(a.lb ** 2 - dmse) - a.lb)
    pd.set_option('display.width', 250)
    print(r.round(3).to_string())
    print(f'pooled g_best {pooled:,.0f} s^2 over {int(m.best.sum())} rows')
    for k, (dmse, d) in out.items():
        print(f'{k:22s}: dMSE {dmse:,.0f} s^2 -> dRMSE {d:+.2f} s on the scored months (LB {a.lb})  [ESTIMATE]')
    r.to_csv(a.match.with_suffix('.extrapolation.csv'))


if __name__ == '__main__':
    main()
