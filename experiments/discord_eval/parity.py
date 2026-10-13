"""2025-vs-2026 parity of the new wake and backlog columns (Piyush Patil's distribution check on
the challenge Discord): the same months (January, July) of the training file vs the ranking file, per column, as
a standardised mean shift (mean_2026 - mean_2025) / sd_2025, pooled and worst airport.
Label-free on both sides.

    python experiments/discord_eval/parity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from prc.config import CACHE
import features as F


def main():
    for which in ('training', 'ranking'):
        F.wake_features(which); F.backlog_features(which)
    tr = pd.concat([F.wake_features('training'), F.backlog_features('training')], axis=1)
    rk = pd.concat([F.wake_features('ranking'), F.backlog_features('ranking')], axis=1)
    mt = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'month'])
    mr = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=['APT', 'MVT_TIME_UTC_mvt'])
    mr['month'] = mr.MVT_TIME_UTC_mvt.dt.month
    print(f"{'column':22s}{'month':>6s}{'mean25':>9s}{'mean26':>9s}{'shift/sd':>10s}{'worst apt':>11s}{'shift':>8s}{'NaN25':>7s}{'NaN26':>7s}")
    for c in tr.columns:
        for m in (1, 7):
            a = tr.loc[mt.month.to_numpy() == m, c]; b = rk.loc[mr.month.to_numpy() == m, c]
            sd = a.std() or 1.0
            sh = {}
            for ap in np.unique(mt.APT.astype(str)):
                aa = a[mt.APT.astype(str).to_numpy()[mt.month.to_numpy() == m] == ap]
                bb = b[mr.APT.astype(str).to_numpy()[mr.month.to_numpy() == m] == ap]
                sh[ap] = (bb.mean() - aa.mean()) / (aa.std() or 1.0)
            w = max(sh, key=lambda k: abs(sh[k]) if np.isfinite(sh[k]) else -1)
            print(f'{c:22s}{m:6d}{a.mean():9.2f}{b.mean():9.2f}{(b.mean() - a.mean()) / sd:10.3f}{w:>11s}{sh[w]:8.3f}'
                  f'{a.isna().mean():7.3f}{b.isna().mean():7.3f}')


if __name__ == '__main__':
    main()
