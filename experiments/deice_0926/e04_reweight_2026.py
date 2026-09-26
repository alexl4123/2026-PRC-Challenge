"""De-icing / disruption indicator, step 4 [E]: re-weight the measured 2025 harness effect to the 2026 mix.

The 2026 ranking file has ~8x more rows in hot remote-rest hours than 2025 (e03). The 12-month harness average
therefore under-weights exactly the rows the indicator acts on. This step takes the 2025 per-row change in squared
error (arm - control, from e02's cached predictions), averages it within label-free cells, and applies the cell
means to the 2026 cell counts:

  cells = airport group (EHAM / EDDM / LSZH / other covered / uncovered LFPG-LTFM) x bin of dz_rr_sh_h
          (NaN, 0, (0, .05], (.05, .15], > .15)
  dSSE_2026 = sum_cell  mean_2025(d e^2 | cell) * n_2026(cell)
  live delta = sqrt(251.12^2 + dSSE_2026 / N_2026) - 251.12        (assumes the rest of the live SSE is unchanged)

Uncertainty: airport-day block bootstrap of the 2025 cell means (1,000 draws). Rows with |y| > 12 h (the day-scale
label artefacts, which no feature reaches) are dropped from the cell means in the main estimate; the estimate with
them is printed too. This is an ESTIMATE of the harness effect on the 2026 mix, not a live number; the harness ->
live conversion factor is separate (see the report).

    python experiments/deice_0926/e04_reweight_2026.py [--arm cand]
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np
import pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

OUT = CACHE / 'deice_0926'
SCR = os.environ.get('DISCORD_SCRATCH', '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-'
                     'Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/deice_0926')
LIVE = 251.12


def cells(apt, sh):
    grp = np.where(np.isin(apt, ['EHAM', 'EDDM', 'LSZH']), apt, np.where(np.isin(apt, ['LFPG', 'LTFM']), 'LFPG/LTFM', 'other'))
    b = np.select([np.isnan(sh), sh == 0, sh <= .05, sh <= .15], ['nan', '0', '(0,.05]', '(.05,.15]'], '>.15')
    return pd.Series(grp).astype(str) + '|' + pd.Series(b).astype(str)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='cand'); ap.add_argument('--col', default='dz_rr_sh_h')
    a = ap.parse_args()
    b = np.load(CACHE / 'v11' / 'base.npz'); y, month = b['y'], b['month']
    p = np.load(f'{SCR}/deice__{a.arm}.npy').astype(float); c = np.load(f'{SCR}/deice__control.npy').astype(float)
    d = (y - p) ** 2 - (y - c) ** 2
    it = pd.read_parquet(OUT / 'ind_training.parquet', columns=[a.col]); ir = pd.read_parquet(OUT / 'ind_ranking.parquet', columns=[a.col])
    rt = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
    rr = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet', columns=['apt', 'month'])
    ct = cells(rt.apt.astype(str).to_numpy(), it[a.col].to_numpy(float)).to_numpy()
    cr = cells(rr.apt.astype(str).to_numpy(), ir[a.col].to_numpy(float))
    n26 = cr.value_counts()
    N26 = len(cr)
    blk = pd.factorize(rt.apt.astype(str) + '_' + rt.date_loc.astype(str))[0]
    for label, keep in (('main (|y| <= 12 h)', np.abs(y) <= 43_200), ('with day-scale rows', np.ones(len(y), bool))):
        k = keep & np.isfinite(d)
        cid, cu = pd.factorize(ct[k])
        tab = pd.DataFrame({'cell': cu, 'n_2025': np.bincount(cid), 'dmse_2025': np.bincount(cid, d[k]) / np.bincount(cid)})
        tab['n_2026'] = tab.cell.map(n26).fillna(0).astype(int)
        tab['dsse_2026'] = tab.dmse_2025 * tab.n_2026
        dsse = tab.dsse_2026.sum()
        live = np.sqrt(LIVE ** 2 + dsse / N26) - LIVE
        # block bootstrap over 2025 airport-days
        bid, bu = pd.factorize(blk[k]); nb = len(bu); nc = len(cu)
        key = bid * nc + cid
        S = np.bincount(key, d[k], minlength=nb * nc).reshape(nb, nc); Nn = np.bincount(key, minlength=nb * nc).reshape(nb, nc)
        w26 = tab.n_2026.to_numpy(float)
        # signal-only: keep only the hot-hour cells (share > 5%), where the indicator carries information; every
        # other cell is set to 0 (its 2025 change is re-fit variation, e05)
        hot = tab.cell.str.endswith('(.05,.15]') | tab.cell.str.endswith('>.15')
        w26s = np.where(hot, w26, 0.0)
        live_s = np.sqrt(LIVE ** 2 + (tab.dmse_2025.to_numpy() * w26s).sum() / N26) - LIVE
        rng = np.random.default_rng(0); draws = np.empty(1000); draws_s = np.empty(1000)
        for i in range(1000):
            wb = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            s, n = wb @ S, wb @ Nn
            m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
            draws[i] = np.sqrt(LIVE ** 2 + (m * w26).sum() / N26) - LIVE
            draws_s[i] = np.sqrt(LIVE ** 2 + (m * w26s).sum() / N26) - LIVE
        missing = int(n26[~n26.index.isin(cu)].sum())
        print(f'\n[{a.arm}, {a.col}] {label}: 2026-mix harness delta {live:+.3f} s  95% block CI '
              f'[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}]  (2026 rows in cells unseen in 2025: {missing})')
        print(f'   signal-only (hot-hour cells, {int(w26s.sum()):,} 2026 rows): {live_s:+.3f} s  95% block CI '
              f'[{np.percentile(draws_s, 2.5):+.3f}, {np.percentile(draws_s, 97.5):+.3f}]')
        print(tab.sort_values('cell').round(2).to_string(index=False))
        if label.startswith('main'):
            tab.to_csv(OUT / f'e04_reweight_{a.arm}.csv', index=False)
    print(f'peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
