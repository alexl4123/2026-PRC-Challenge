import numpy as np, pandas as pd
pd.set_option('display.width', 250)
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
m = pd.read_parquet(H + 'h22_union25.parquet'); m6 = pd.read_parquet(H + 'h22_union26.parquet')
for nm, x in (('2025', m), ('2026', m6)):
    tw = x[~x.onesided]
    cls = np.where(tw.stamp_excl, 'os in window', np.where(tw.off_sched < tw.lo, 'os < lo', 'os > hi'))
    if nm == '2025':
        print(pd.crosstab(cls, tw.side))
        print('  os>hi rows by APT:', tw[cls == 'os > hi'].APT.value_counts().to_dict())
    else:
        ins = (tw.f > tw.lo) & (tw.f < tw.hi)
        t = tw[ins].assign(cls=cls[ins.to_numpy()], move=np.maximum(tw.lo, 0)[ins] - tw.f[ins])
        print(t[['APT', 'FLIGHT_mvt', 'cls', 'f', 'lo', 'hi', 'off_sched', 'move']].sort_values('cls').to_string())
        for c in ['os < lo', 'os > hi']:
            s = t[t.cls == c].move; print(f'  2026 {c}: rows {len(s)}, sum sq {(s ** 2).sum():.4g}, bound {np.sqrt(235.3996 ** 2 - (s ** 2).sum() / 344841) - 235.3996:+.4f}')
