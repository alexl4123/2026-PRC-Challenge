import pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
c = ['PHASE_mvt','APT','STAND_mvt','AIRCRAFT_OPERATOR_flt','ADES_mvt','AIRCRAFT_TYPE_mvt','MVT_TIME_UTC_mvt']
tr = pd.read_parquet(R+'cache/training.parquet', columns=c, filters=[('PHASE_mvt','==','DEP')])
rk = pd.read_parquet(R+'cache_final/ranking.parquet', columns=c, filters=[('PHASE_mvt','==','DEP')])
rk['mo'] = rk.MVT_TIME_UTC_mvt.dt.month
for col, apt_wise in [('STAND_mvt', True), ('AIRCRAFT_OPERATOR_flt', False), ('ADES_mvt', False), ('AIRCRAFT_TYPE_mvt', False)]:
    if apt_wise:
        seen = set(zip(tr.APT.astype(str), tr[col].astype(str))); u = pd.Series([(a, s) not in seen for a, s in zip(rk.APT.astype(str), rk[col].astype(str))], index=rk.index)
    else:
        u = ~rk[col].astype(str).isin(set(tr[col].astype(str)))
    u &= rk[col].notna()
    print(col, 'unseen rows by month', u.groupby(rk.mo).sum().to_dict(), 'total', int(u.sum()))
    if col == 'STAND_mvt':
        print('  EDDM', int(u[rk.APT.astype(str) == 'EDDM'].sum()), 'EDDF', int(u[rk.APT.astype(str) == 'EDDF'].sum()))
    if col == 'AIRCRAFT_OPERATOR_flt':
        print('  top unseen ops', rk[u].groupby(rk[u][col].astype(str)).size().sort_values(ascending=False).head(6).to_dict())
