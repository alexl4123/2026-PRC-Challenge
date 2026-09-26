"""Drivers 09-26, step 6: exposure check for the P branch without ADS-B, on the 2026 ranking set (label-free).

v17 serves zone P with TabICL p x off_sched + (1 - p) x (0.5 W2-mix + 0.5 W2-nn bulk, NO ADS-B columns). The
'ADS-B bulk inside P' arm failed its 2025 per-month rule (+0.04 s in July) and was not shipped. 2025 has few
covered P rows; 2026 January has 3x more P rows (EHAM disruption) and better coverage. This measures, per airport
and route, how far the served prediction is from the flight's own ADS-B-implied taxi time (MVT - adsb_off_best),
for v16, v17 and the unshipped ADS-B P branch, and the 2025 reference |label - adsb taxi| on the same kind of rows.
Nothing here uses a 2026 label (there are none).

    python experiments/drivers_0926/d06_p_adsb_2026.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS

OUT = CACHE / 'drivers_0926'
pd.set_option('display.width', 220)


def seeds(folder, fam, ids):
    assert np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids)
    return np.mean([np.load(folder / f'{fam}__seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)


def main():
    k = pd.read_parquet(OUT / 'rows_ranking.parquet'); r = pd.read_parquet(OUT / 'rows_training.parquet')
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, et, in_P = c['ids'], c['et'], c['in_P'].astype(bool)
    assert (in_P == (k.route == 'P').to_numpy()).all()
    s16 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v16.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy(float)
    s17 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v17.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt'].reindex(ids).to_numpy(float)
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
    pb_w2 = 0.5 * seeds(CACHE / 'w2_final', 'mix', ids) + 0.5 * seeds(CACHE / 'w2_final', 'nn', ids)
    pb_ad = 0.5 * seeds(CACHE / 'adsb2_final', 'mix', ids) + 0.5 * seeds(CACHE / 'adsb2_final', 'nn', ids)
    alt = s17.copy()
    alt[rows] = s17[rows] + (np.clip(p * et[rows] + (1 - p) * pb_ad[rows], 0, None) - np.clip(p * et[rows] + (1 - p) * pb_w2[rows], 0, None))
    pp = np.full(len(k), np.nan); pp[rows] = p
    k['v16'], k['v17'], k['alt'], k['p'] = s16, s17, alt, pp
    cov = k.adsb_off.to_numpy() & (k.adsb_taxi.to_numpy() > 0) & (k.adsb_taxi.to_numpy() < 7200)
    g = k[cov].assign(d16=lambda x: (x.v16 - x.adsb_taxi).abs(), d17=lambda x: (x.v17 - x.adsb_taxi).abs(),
                      dalt=lambda x: (x.alt - x.adsb_taxi).abs())
    t = g.groupby(['route', 'apt'], observed=True).agg(n=('d17', 'size'), med_v16=('d16', 'median'), med_v17=('d17', 'median'),
                                                       med_altP=('dalt', 'median'), p_mean=('p', 'mean'))
    print('2026 rows with an ADS-B off-block (0 < taxi < 2 h): median |served - ADS-B taxi| (s)')
    print(t[t.n >= 5].round(1).to_string())
    # total P rows per airport-month and covered share
    kp = k[k.route == 'P']
    print('\n2026 zone P rows by airport x month (covered = ADS-B off-block present):')
    print(kp.groupby(['apt', 'month'], observed=True).agg(n=('p', 'size'), covered=('adsb_off', 'sum'), p_mean=('p', 'mean')).query('n > 0').to_string())
    # 2025 reference: label vs ADS-B taxi in P, real pushbacks
    rp = r[(r.route == 'P') & r.adsb_off & (r.y <= 7200)]
    print(f'\n2025 zone P real pushbacks with an ADS-B off-block: {len(rp)} rows; median |y - adsb taxi| '
          f'{np.median(np.abs(rp.y - rp.adsb_taxi)):.0f} s; median |v17 OOF - y| {np.median(np.abs(rp.pred - rp.y)):.0f} s')
    # shift size of the unshipped alternative on 2026 P rows
    d = (alt - s17)[rows]
    print(f'unshipped ADS-B P branch vs v17 on 2026 P rows: mean {d.mean():+.1f} s, |d| median {np.median(np.abs(d)):.0f}, '
          f'rows |d| > 300: {int((np.abs(d) > 300).sum())}')
    t.to_csv(OUT / 't15_2026_served_vs_adsb.csv')


if __name__ == '__main__':
    main()
