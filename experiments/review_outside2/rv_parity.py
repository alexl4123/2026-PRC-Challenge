from common import *
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_outside2/'
tr = pd.read_parquet(f'{D}/rv_stand_cont.parquet'); rk = pd.read_parquet(O + 'c4_stand_ranking.parquet')
dt = load_dep(['APT', 'MVT_TIME_UTC_mvt']); dr = load_dep(['APT', 'MVT_TIME_UTC_mvt'], 'ranking')
mt = pd.to_datetime(dt.MVT_TIME_UTC_mvt, unit='s').dt.month.to_numpy(); mr = pd.to_datetime(dr.MVT_TIME_UTC_mvt, unit='s').dt.month.to_numpy()
rows = []
for nm, F, mo, ap in [('2025', tr, mt, S_(dt.APT)), ('2026', rk, mr, S_(dr.APT))]:
    for mm in [1, 7]:
        k = mo == mm
        f = F[k]; ok = f.sl_to.abs() < 30 * 86400
        proxy = ((f.sl_after_sched > 0) & (f.sl_same_typ == 0))[ok]       # last arrival parked after our SCHED and is another type
        rows.append(dict(set=nm, m=mm, nextocc_proxy=proxy.mean(), sl_to_med=f.sl_to[ok].median(), sa_after_med=f.sa_after_to[f.sa_after_to.abs() < 30*86400].median(),
                         sn_same_typ=f.sn_same_typ.mean(), sl_same_op=f.sl_same_op.mean(),
                         lirf_proxy=((f.sl_after_sched > 0) & (f.sl_same_typ == 0))[ok & (ap[k] == 'LIRF')].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
