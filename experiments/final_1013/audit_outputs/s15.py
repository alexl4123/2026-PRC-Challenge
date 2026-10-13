import pandas as pd
rk = pd.read_parquet('rk.parquet'); rk['dd'] = rk.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')
for apt, m in (('EDDM', 1), ('EDDM', 2), ('LEMD', 2), ('EDDF', 1), ('EDDF', 2), ('LSZH', 1), ('EHAM', 1)):
    x = rk[(rk.APT == apt) & (rk.month == m)]; s = x.groupby('dd').P.sum(); dsh = x.groupby('dd').D.sum()
    print(apt, m, 'P total', int(s.sum()), 'top days', s.sort_values().tail(4).to_dict(), 'days with P>0', int((s > 0).sum()), '| D total', int(dsh.sum()), 'top D days', dsh.sort_values().tail(3).to_dict())
