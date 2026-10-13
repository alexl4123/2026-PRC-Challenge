"""Repeated departure rows of one flight (same APT, FLIGHT_mvt, SCHED): is the later row's BLOCK after the earlier row's take-off?"""
import numpy as np, pandas as pd
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; yy = V['y']; month = V['month']; P = V['P']
for yr in ('tr', 'ra'):
    d = pd.read_parquet(H + f'{yr}_dep.parquet', columns=['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'SCHED', 'MVT', 'LOBT', 'STAND_mvt', 'RUNWAY_mvt', 'FLIGHT_RULE_mvt'] + (['BLOCK'] if yr == 'tr' else []))
    d['i'] = np.arange(len(d)); d['APT'] = d.APT.astype(str)
    if yr == 'tr': d['y'] = d.MVT - d.BLOCK; d['f'] = f; d['P'] = P; d['month'] = month
    else:
        v = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').rename(columns={'TAXITIME_SEC_mvt': 'f'})
        d = d.merge(v, on='MVT_ID_mvt', how='left')
    g = d.dropna(subset=['FLIGHT_mvt']).sort_values('MVT')
    g['n'] = g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True).MVT.transform('size')
    g = g[g.n > 1].copy()
    g['rank'] = g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True).MVT.rank(method='first')
    g['prev_mvt'] = g.groupby(['APT', 'FLIGHT_mvt', 'SCHED'], observed=True).MVT.shift(1)
    g['ub'] = g.MVT - g.prev_mvt          # y of a later row <= MVT - previous take-off  (if BLOCK_later >= MVT_earlier)
    g['off_sched'] = g.MVT - g.SCHED; g['nm'] = g.LOBT.notna()
    later = g[g['rank'] > 1]
    print(f'\n== {yr}: rows in repeated groups {len(g)}, later rows {len(later)}  (with LOBT {later.nm.sum()})')
    if yr == 'tr':
        later = later.assign(ok=later.y <= later.ub + 12)
        print(' later-row y <= MVT - prev MVT (+12):', later.ok.sum(), 'of', len(later))
        print(later[~later.ok][['APT', 'FLIGHT_mvt', 'y', 'ub', 'off_sched', 'nm', 'f']].to_string())
        print(' later rows: y vs off_sched (|y-off_sched|<=6):', ((later.y - later.off_sched).abs() <= 6).sum(), '  f > ub:', (later.f > later.ub).sum())
        k = later[later.f > later.ub + 12]
        print(k[['APT', 'FLIGHT_mvt', 'y', 'f', 'ub', 'off_sched', 'nm', 'P', 'month']].to_string())
        # earlier rows: y vs the later MVT
        first = g[g['rank'] == 1]
        print(' first rows y quantiles', first.y.quantile([0, .5, .9, .99, 1]).round().to_dict(), ' later rows y quantiles', later.y.quantile([0, .5, .9, .99, 1]).round().to_dict())
    else:
        k = later[later.f > later.ub]
        print(' v28 later rows with f > MVT - prev MVT:', len(k)); print(k[['APT', 'FLIGHT_mvt', 'f', 'ub', 'off_sched', 'nm']].to_string())
