"""Drivers 09-26, step 3: (a) where the 2026 error is EXPECTED to live (2025 per-cell MSE x 2026 cell counts, label-free
on the 2026 side); (b) the exact-zero spike of SCHED - BLOCK (a stamping signature that does not depend on a threshold);
(c) ADS-B status mix 2025 vs 2026 per airport incl. rows whose archive day failed; (d) label-free disruption-day
indicators, 2026 vs the 2025 distribution; (e) what the ADS-B gain per status implies for an airport without coverage.

    python experiments/drivers_0926/d03_extra.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb
from d02_decompose import mechanism, adsb_status

OUT = CACHE / 'drivers_0926'
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)


def main():
    t0 = time.time(); S2 = {}
    r = pd.read_parquet(OUT / 'rows_training.parquet'); k = pd.read_parquet(OUT / 'rows_ranking.parquet')
    y = r.y.to_numpy(); e2 = (y - r.pred.to_numpy()) ** 2
    r['mech'] = mechanism(r); r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
    day = r.mech.to_numpy() == 'day'

    # (a) expected 2026 SSE by airport x route x ADS-B status (2025 12-month cell MSE, day-scale rows excluded)
    def cell(x):
        sob = pd.cut(x.off_sched, [-np.inf, 0, 900, 1800, 3600, 7200, np.inf], labels=False).fillna(-1).astype(int).astype(str)
        return x.apt.astype(str) + '|' + x.route.astype(str) + '|' + x.adsb.astype(str), sob
    c25, b25 = cell(r); c26, b26 = cell(k)
    fine25 = (c25 + '|' + b25).to_numpy(); fine26 = (c26 + '|' + b26).to_numpy()
    mse = pd.Series(e2[~day]).groupby(fine25[~day]).mean(); msec = pd.Series(e2[~day]).groupby(c25.to_numpy()[~day]).mean()
    m26 = pd.Series(fine26).map(mse); miss = m26.isna().to_numpy(); m26[miss] = pd.Series(c26.to_numpy()[miss]).map(msec).to_numpy()
    m26 = m26.fillna(m26.mean()).to_numpy()
    exp = pd.DataFrame({'cell': c26.to_numpy(), 'e': m26, 'apt': k.apt.astype(str).to_numpy(), 'route': k.route.astype(str).to_numpy(),
                        'adsb': k.adsb.to_numpy()})
    tot = exp.e.sum()
    ta = exp.groupby(['apt', 'route', 'adsb']).agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot),
                                                    rmse=('e', lambda v: np.sqrt(v.mean()))).sort_values('sse_pct', ascending=False)
    ta.to_csv(OUT / 't10_expected_2026_cells.csv')
    print(f'(a) EXPECTED 2026 error (2025 12m cell MSE, day-scale rows excluded): RMSE {np.sqrt(exp.e.mean()):.1f}; top cells')
    print(ta.head(20).round(2).to_string())
    tb = exp.groupby('apt').agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot), rmse=('e', lambda v: np.sqrt(v.mean())))
    # the same, 2025 Jan+Jul own mix, for the side-by-side
    S = r.month.isin([1, 7]).to_numpy() & ~day
    own = pd.DataFrame({'apt': r.apt.astype(str).to_numpy()[S], 'e': e2[S]}).groupby('apt').agg(
        rows25=('e', 'size'), sse_pct25=('e', lambda v: 100 * v.sum() / e2[S].sum()), rmse25=('e', lambda v: np.sqrt(v.mean())))
    tb = tb.join(own); tb.to_csv(OUT / 't10b_expected_2026_airports.csv'); print('\n  by airport (2026 expected vs 2025 Jan+Jul actual, day-scale excluded)'); print(tb.round(2).to_string())
    S2['expected_2026_rmse_noday'] = float(np.sqrt(exp.e.mean()))

    # (b) exact-zero spike of SCHED - BLOCK = y - off_sched (both on whole minutes?)
    d = (y - r.off_sched.to_numpy())
    fin = np.isfinite(d)
    print('\n(b) share of rows with (y - off_sched) a multiple of 60 s:', f'{np.mean(np.mod(d[fin], 60) == 0):.3f}')
    rows = []
    for a in sorted(r.apt.astype(str).unique()):
        m = fin & (r.apt.astype(str).to_numpy() == a) & (r.off_sched.to_numpy() > 1200) & (y <= 7200)
        dd = np.round(d[m]).astype(int)
        z = np.mean(np.abs(dd) <= 30); nb = np.mean((np.abs(dd) > 30) & (np.abs(dd) <= 150)) / 4   # 4 neighbouring minute bins
        rows.append((a, m.sum(), 100 * z, 100 * nb, 100 * (z - nb)))
    tz = pd.DataFrame(rows, columns=['apt', 'rows_delayed', 'pct_at_0', 'pct_per_nbr_min', 'excess_pct']).set_index('apt')
    tz.to_csv(OUT / 't11_zero_spike.csv')
    print('   delayed flights (off_sched > 20 min, y <= 2 h): % with |SCHED-BLOCK| <= 30 s vs mean of the 4 neighbouring minute bins')
    print(tz.round(2).to_string())

    # (c) ADS-B status mix per airport and month, 2025 vs 2026; archive-day failures
    def mix(x, tag):
        g = pd.crosstab([x.apt.astype(str), x.month], x.adsb, normalize='index') * 100
        g.columns = [f'{tag} {c}' for c in g.columns]; return g
    tm = mix(r[r.month.isin([1, 7])], '2025').join(mix(k, '2026'))
    tm.to_csv(OUT / 't12_adsb_mix.csv'); print('\n(c) ADS-B status % by airport x month'); print(tm.round(1).to_string())

    # (d) label-free disruption indicators per airport-day: median off_sched, share without NM record
    def daytab(x):
        return x.groupby([x.apt.astype(str), x.date_loc]).agg(n=('off_sched', 'size'), so_med=('off_sched', 'median'),
                                                               shareD=('route', lambda v: (v != 'ABC').mean()))
    d25, d26 = daytab(r), daytab(k)
    thr = d25.groupby(level=0).so_med.quantile(0.99); thrD = d25.groupby(level=0).shareD.quantile(0.99)
    d26['hot_so'] = d26.so_med.to_numpy() > d26.index.get_level_values(0).map(thr).to_numpy()
    d26['hot_D'] = d26.shareD.to_numpy() > d26.index.get_level_values(0).map(thrD).to_numpy()
    hot = d26[d26.hot_so | d26.hot_D]
    hot.to_csv(OUT / 't13_2026_hot_days.csv')
    print(f'\n(d) 2026 airport-days above the 2025 per-airport 99th percentile of median off_sched or share-no-NM: {len(hot)} of {len(d26)}')
    print(hot.round(3).to_string())
    # 2025 ordinary-row MSE on its own hot days vs other days
    gd = pd.read_csv(OUT / 't6_airport_days.csv').set_index(['apt', 'date'])
    gd['hot'] = (gd.so_med.to_numpy() > gd.index.get_level_values(0).map(thr).to_numpy()) | \
                (gd.shareD.to_numpy() > gd.index.get_level_values(0).map(thrD).to_numpy())
    print('   2025 ordinary rows: MSE ratio to airport mean on hot days', gd[gd.hot].ratio.median().round(2),
          'mean', gd[gd.hot].ratio.mean().round(2), '| other days median', gd[~gd.hot].ratio.median().round(2),
          '| hot days share of ordinary SSE', round(100 * gd[gd.hot].sse.sum() / gd.sse.sum(), 1), '% on', int(gd.hot.sum()), 'days')

    # (e) per-status ADS-B gain -> what coverage would buy at an uncovered airport (estimate)
    t3 = pd.read_csv(OUT / 't3_adsb_apt.csv').set_index(['apt', 'adsb'])
    print('\n(e) relative MSE gain v15->v17 by ADS-B status (covered airports, ordinary A/B/C):')
    print(t3.gain_pct.unstack().round(1).to_string())
    json.dump(S2, open(OUT / 'summary_d03.json', 'w'), indent=1)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
