"""Drivers 09-26, step 7: translate driver cells into LIVE-score upside (estimates, clearly conditional).

Basis: the expected 2026 per-row MSE from d03 (2025 12-month cell MSE on the 2026 cell mix, day-scale rows excluded)
is taken as the reachable part of the live MSE; live MSE = 251.12^2. A scenario that lowers the expected SSE of some
2026 rows by dS lowers the live MSE by dS / N (the unexplained ~25% residual is assumed untouched).
Scenario gains per ADS-B status are the MEASURED v15 -> v17 relative MSE reductions on ordinary rows (t3_adsb_apt.csv).

    python experiments/drivers_0926/d07_upside.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from d02_decompose import mechanism, adsb_status

OUT = CACHE / 'drivers_0926'
LIVE = 251.1243
N = 344841


def main():
    r = pd.read_parquet(OUT / 'rows_training.parquet'); k = pd.read_parquet(OUT / 'rows_ranking.parquet')
    y = r.y.to_numpy(); e2 = (y - r.pred.to_numpy()) ** 2
    r['mech'] = mechanism(r); r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
    day = r.mech.to_numpy() == 'day'
    sob = lambda x: pd.cut(x.off_sched, [-np.inf, 0, 900, 1800, 3600, 7200, np.inf], labels=False).fillna(-1).astype(int).astype(str)
    key = lambda x, st: (x.apt.astype(str) + '|' + x.route.astype(str) + '|' + st + '|' + sob(x)).to_numpy()
    mse = pd.Series(e2[~day]).groupby(key(r, r.adsb.astype(str))[~day]).mean()
    coarse = pd.Series(e2[~day]).groupby((r.apt.astype(str) + '|' + r.route.astype(str)).to_numpy()[~day]).mean()

    def expected(st):
        m = pd.Series(key(k, st)).map(mse)
        miss = m.isna().to_numpy()
        m[miss] = pd.Series((k.apt.astype(str) + '|' + k.route.astype(str)).to_numpy()[miss]).map(coarse).to_numpy()
        return m.to_numpy()
    base = expected(k.adsb.astype(str))
    live_mse = LIVE ** 2
    print(f'expected day-free 2026 RMSE {np.sqrt(base.mean()):.1f}; live {LIVE}; unexplained MSE {live_mse - base.mean():.0f}')
    t3 = pd.read_csv(OUT / 't3_adsb_apt.csv').set_index(['apt', 'adsb'])
    g_off = t3.xs('off-block seen', level='adsb').gain_pct / 100; g_gr = t3.xs('ground only', level='adsb').gain_pct / 100
    weak = ['LEBL', 'LEMD', 'EGLL']                  # airports whose labels follow the aircraft less tightly
    G_OFF, G_GR = g_off[weak].mean(), g_gr[weak].mean()
    print(f'gain assumptions (mean of LEBL/LEMD/EGLL): off-block seen {G_OFF:.2f}, ground only {G_GR:.2f}')

    def report(name, dS):
        new = np.sqrt(live_mse - dS.sum() / N)
        print(f'  {name:78s} live {LIVE:.2f} -> {new:.2f}  ({new - LIVE:+.2f} s)')
    ordA = (k.route == 'ABC').to_numpy()
    rng = np.random.default_rng(0)
    for apt, c_off, c_gr in (('LTFM', 0.186, 0.665), ('LTFM', 0.50, 0.40), ('LFPG', 0.186, 0.665), ('LFPG', 0.50, 0.40)):
        m = ordA & (k.apt == apt).to_numpy()
        u = rng.random(m.sum())
        # rows currently in the airport's no-ground / ground-only cells move to covered states (probabilistic mix)
        frac = np.where(u < c_off, G_OFF, np.where(u < c_off + c_gr, G_GR, 0.0))
        cur_gain = np.where(k.adsb.to_numpy()[m] == 'ground only', 0.13 if apt == 'LFPG' else 0.0, 0.0)   # LFPG ground-only already has its 13%
        dS = base[m] * np.clip(frac - cur_gain, 0, None)
        report(f'{apt}: A/B/C coverage {int(100*c_off)}% off-block + {int(100*c_gr)}% ground-only (LEMD-2026-like if 19/66)', dS)
    for apt in ('LTFM', 'LFPG', 'LIRF', 'EGLL'):
        m = ordA & (k.apt == apt).to_numpy()
        floor = 4316.0
        report(f'{apt}: ORACLE all A/B/C rows at the ADS-B-observed floor (66 s)', np.clip(base[m] - floor, 0, None))
    mP = (k.route == 'P').to_numpy() & (k.apt == 'LIRF').to_numpy()
    report('LIRF zone P: ORACLE (expected P error -> bulk level 500^2)', np.clip(base[mP] - 500 ** 2, 0, None))
    report('LIRF zone P: 20% of that oracle', 0.2 * np.clip(base[mP] - 500 ** 2, 0, None))
    mD = (k.route == 'D').to_numpy() & (k.apt == 'LIRF').to_numpy()
    report('LIRF route D (not P): ORACLE -> 500^2', np.clip(base[mD] - 500 ** 2, 0, None))
    # Rome bulk stamping: share of LIRF A/B/C expected SSE that sits on stamped-signature rows in 2025
    lirfA = (r.apt == 'LIRF').to_numpy() & (r.route == 'ABC').to_numpy() & ~day
    sig = lirfA & (r.mech.to_numpy() == 'schedB')
    share = e2[sig].sum() / e2[lirfA].sum()
    mA = ordA & (k.apt == 'LIRF').to_numpy()
    print(f'  (LIRF A/B/C: share of 2025 SSE on BLOCK=SCHED-signature rows = {share:.2f})')
    report('LIRF A/B/C: ORACLE on the stamping-signature share of expected SSE', share * base[mA])
    report('LIRF A/B/C: 10% of that oracle', 0.1 * share * base[mA])
    # disruption: hot days (d03 t13) -- ordinary MSE ratio 2.6 in 2025; scenario: ratio -> 2.0 on 2026 hot days
    hot = pd.read_csv(OUT / 't13_2026_hot_days.csv')
    hk = set(zip(hot.apt.astype(str), hot.date_loc))
    mh = ordA & np.array([(a, d) in hk for a, d in zip(k.apt.astype(str), k.date_loc)])
    report(f'2026 hot days ({mh.sum():,} A/B/C rows): error x2.6 assumed -> x2.0 (i.e. -23% on those rows)', base[mh] * 2.6 * (1 - 2.0 / 2.6) - 0 * base[mh])
    # everything ordinary at covered airports down 5% (e.g. a better nn encoding / more ADS-B detail)
    cov_apts = ['EDDF', 'EDDM', 'EHAM', 'LSZH', 'LEBL', 'LEMD', 'EGLL']
    mc = ordA & k.apt.isin(cov_apts).to_numpy()
    report('covered airports A/B/C: -5% MSE (generic model improvement)', 0.05 * base[mc])
    report('ALL A/B/C rows: -5% MSE (generic model improvement)', 0.05 * base[ordA])


if __name__ == '__main__':
    main()
