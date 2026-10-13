"""Per-subset bounds: BLOCK-LOBT (route A/B/C) and BLOCK-SCHED, y (route D) by airport / flight rule / market segment / flight type."""
import numpy as np, pandas as pd
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
d = pd.read_parquet(H + 'tr_dep.parquet', columns=['APT', 'FLIGHT_RULE_mvt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt', 'FLIGHT_RULE_flt', 'MVT', 'BLOCK', 'SCHED', 'LOBT', 'EOBT1', 'IOBT', 'AOBT3'])
for c in ['APT', 'FLIGHT_RULE_mvt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt', 'FLIGHT_RULE_flt']: d[c] = d[c].astype(str)
d['y'] = d.MVT - d.BLOCK; nm = d.LOBT.notna()
def tab(x, val, by):
    g = x.groupby(by)[val]
    t = pd.DataFrame({'n': g.size(), 'min': g.min(), 'q1e-3': g.quantile(.001), 'q.999': g.quantile(.999), 'max': g.max(),
                      'n<=min+30': g.apply(lambda s: (s <= s.min() + 30).sum()), 'n>=max-30': g.apply(lambda s: (s >= s.max() - 30).sum())})
    return t.round(0)
x = d[nm].assign(bl=d.BLOCK - d.LOBT, be=d.BLOCK - d.EOBT1, bi=d.BLOCK - d.IOBT)
print('route A/B/C: BLOCK - LOBT by APT'); print(tab(x, 'bl', 'APT').to_string())
print('route A/B/C: BLOCK - LOBT by MARKET_SEGMENT'); print(tab(x, 'bl', 'MARKET_SEGMENT_flt').to_string())
print('route A/B/C: BLOCK - LOBT by FLIGHT_TYPE'); print(tab(x, 'bl', 'FLIGHT_TYPE_flt').to_string())
print('route A/B/C: BLOCK - LOBT by FLIGHT_RULE_flt'); print(tab(x, 'bl', 'FLIGHT_RULE_flt').to_string())
print('route A/B/C: y by APT'); print(tab(x, 'y', 'APT').to_string())
z = d[~nm].assign(bs=d.BLOCK - d.SCHED)
print('route D: BLOCK - SCHED by APT x FLIGHT_RULE_mvt'); print(tab(z, 'bs', ['APT', 'FLIGHT_RULE_mvt']).to_string())
print('route D: y by APT x FLIGHT_RULE_mvt'); print(tab(z, 'y', ['APT', 'FLIGHT_RULE_mvt']).to_string())
