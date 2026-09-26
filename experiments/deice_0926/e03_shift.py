"""De-icing / disruption indicator, step 3 (label-free): how do the indicator columns differ between 2025 (training)
and the Jan + Jul 2026 ranking file? Reads only cache/deice_0926/ind_*.parquet and the airport / month / local-date
columns of cache/drivers_0926/rows_*.parquet -- no label, no prediction.

Outputs cache/deice_0926/e03_shift_apt_month.csv, e03_quantiles.csv, e03_hot_days_{2025,2026}.csv

    python experiments/deice_0926/e03_shift.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

OUT = CACHE / 'deice_0926'
COLS = ['dz_ncov_h', 'dz_rr_sh_h', 'dz_rr_n_h', 'dz_ls_sh_h', 'dz_rest_h', 'dz_ncov_d', 'dz_rr_sh_d', 'dz_rr_n_d',
        'dz_ls_sh_d', 'dz_taxi_h']


def frame(which, year):
    X = pd.read_parquet(OUT / f'ind_{which}.parquet')
    r = pd.read_parquet(CACHE / 'drivers_0926' / f'rows_{which}.parquet', columns=['apt', 'month', 'date_loc'])
    X['apt'] = r.apt.astype(str).to_numpy(); X['month'] = r.month.to_numpy(); X['date_loc'] = r.date_loc.to_numpy()
    X['year'] = year
    return X


def main():
    tr, rk = frame('training', 2025), frame('ranking', 2026)
    A = pd.concat([tr[tr.month.isin([1, 7])], rk], ignore_index=True)
    g = A.groupby(['apt', 'month', 'year'])
    t = pd.DataFrame({
        'rows': g.size(),
        'own_cov': g.own_cov.mean(),
        'own_rr_per_cov': g.own_rr.sum() / g.own_cov.sum(),
        'own_ls_per_cov': g.own_ls.sum() / g.own_cov.sum(),
        'rr_sh_h_mean': g.dz_rr_sh_h.mean(),
        'rows_rr_sh_h>.05': g.dz_rr_sh_h.apply(lambda s: (s > .05).mean()),
        'rows_rr_sh_h>.15': g.dz_rr_sh_h.apply(lambda s: (s > .15).mean()),
        'ls_sh_h_mean': g.dz_ls_sh_h.mean(),
        'rr_sh_d_max': g.dz_rr_sh_d.max(),
    }).unstack('year')
    t.columns = [f'{a}_{b}' for a, b in t.columns]
    t.to_csv(OUT / 'e03_shift_apt_month.csv')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(t.round(4).to_string())
    # column quantiles, same calendar month, 2025 vs 2026 (all airports)
    q = []
    for m in (1, 7):
        for c in COLS:
            a, b = tr.loc[tr.month == m, c].dropna(), rk.loc[rk.month == m, c].dropna()
            row = dict(month=m, col=c, nn_2025=len(a), nn_2026=len(b))
            for p in (.5, .9, .99, .999):
                row[f'q{p}_2025'] = a.quantile(p); row[f'q{p}_2026'] = b.quantile(p)
            # share of the month's 2026 rows (non-NaN) above the SAME airport's 2025 12-month maximum
            mx = tr.groupby('apt')[c].max()
            rr_ = rk[(rk.month == m) & rk[c].notna()]
            row['share_2026_above_2025_max_same_apt'] = float((rr_[c].to_numpy() > rr_.apt.map(mx).to_numpy()).mean())
            q.append(row)
    q = pd.DataFrame(q); q.to_csv(OUT / 'e03_quantiles.csv', index=False)
    print(q.round(4).to_string())
    # hot airport-days by the whole-day remote-rest share (label-free)
    for X, y in ((tr, 2025), (rk, 2026)):
        d = X.groupby(['apt', 'date_loc']).agg(rows=('own_rr', 'size'), own_rr=('own_rr', 'sum'), cov=('own_cov', 'sum'))
        d['rr_per_cov'] = d.own_rr / d['cov']
        d = d[(d['cov'] >= 50)].sort_values('rr_per_cov', ascending=False)
        d.head(40).to_csv(OUT / f'e03_hot_days_{y}.csv')
        print(f'\n{y}: airport-days with rr_per_cov > 0.05: {(d.rr_per_cov > .05).sum()}, > 0.15: {(d.rr_per_cov > .15).sum()}'
              f' (of {len(d)} with >= 50 covered departures)')
        print(d.head(20).round(3).to_string())
    print(f'peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
