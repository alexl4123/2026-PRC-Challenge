from common import *
v6, v7, bl = preds()
for ap in ['LFPG', 'EGLL']:
    rows = []
    for which in ['training', 'ranking']:
        d = load_dep(['APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt'], which)
        k = S_(d.APT) == ap
        mon = pd.to_datetime(d.MVT_TIME_UTC_mvt.to_numpy()[k], unit='s').month
        rw = S_(d.RUNWAY_mvt)[k]
        t = pd.crosstab(mon, rw, normalize='index').mul(100).round(1)
        t.index = [f'{"25" if which=="training" else "26"}-{m:02d}' for m in t.index]
        rows.append(t)
    print(f'=== {ap} departure runway share (%) by month'); print(pd.concat(rows).fillna(0).to_string())
# LFPG: v6 OOF bulk residual by runway in July 2025 vs rest
d = load_dep(['APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); k = (S_(d.APT) == 'LFPG') & (y <= 7200)
mon = pd.to_datetime(d.MVT_TIME_UTC_mvt.to_numpy(), unit='s').month
t = pd.DataFrame(dict(m=mon[k], rw=S_(d.RUNWAY_mvt)[k], r=(y - v6)[k], y=y[k])).groupby(['rw', 'm']).agg(n=('r', 'size'), bias=('r', 'mean'), ymean=('y', 'mean'))
print('\n=== LFPG v6 OOF bulk residual mean by runway x month (n>300)'); print(t[t.n > 300].round(0).unstack('m').to_string())
