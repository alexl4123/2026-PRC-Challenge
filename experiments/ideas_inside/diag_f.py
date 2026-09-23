"""Per-flight stamping-propensity features for DEP rows (training + ranking), label-free except
where noted; plus 2025-vs-2026 drift of observable recording behaviour."""
from common import *
H = 3600.0
def feats(which):
    d = load_dep(['APT', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'], which)
    a = pd.read_parquet(f'{D}/{which}_ARR.parquet', columns=['APT', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt'])
    L, SA, BA = a.MVT_TIME_UTC_mvt.to_numpy(), a.SCHED_TIME_UTC_mvt.to_numpy(), a.BLOCK_TIME_UTC_mvt.to_numpy()
    a_st = (np.abs(BA - SA) < 60).astype(float)
    a_det = np.abs(L - SA) > 900                     # stamping detectable
    aapt = S_(a.APT); dapt = S_(d.APT)
    M, S, A3 = d.MVT_TIME_UTC_mvt.to_numpy(), d.SCHED_TIME_UTC_mvt.to_numpy(), d.AOBT_3_flt.to_numpy()
    out = pd.DataFrame(index=np.arange(len(d)))
    # 1. previous arrival at our stand (in-block before our SCHED, within 12 h)
    ea = pd.DataFrame(dict(k=K(aapt, S_(a.STAND_mvt)), t=BA, stp=a_st, det=a_det.astype(float),
                           inb_m_sibt=BA - SA, ldg_m_sibt=L - SA)).dropna(subset=['t']).sort_values('t')
    q = pd.DataFrame(dict(k=K(dapt, S_(d.STAND_mvt)), t=S, i=np.arange(len(d)))).dropna(subset=['t']).sort_values('t')
    m = pd.merge_asof(q, ea, on='t', by='k', direction='backward', tolerance=12 * H).set_index('i').reindex(np.arange(len(d)))
    out['pa_stamped'] = m.stp.to_numpy(); out['pa_det'] = m.det.to_numpy()
    out['pa_inb_m_sibt'] = m.inb_m_sibt.to_numpy(); out['pa_ldg_m_sibt'] = m.ldg_m_sibt.to_numpy()
    # 2. airport-day and operator-day arrival stamping rates among detectable arrivals
    aday = (L // 86400).astype(int); dday = (M // 86400).astype(int)
    g = pd.DataFrame(dict(apt=aapt[a_det], day=aday[a_det], st=a_st[a_det])).groupby(['apt', 'day']).st.agg(['mean', 'size'])
    j = pd.DataFrame(dict(apt=dapt, day=dday)).merge(g.reset_index(), on=['apt', 'day'], how='left')
    out['aday_stamp'] = j['mean'].to_numpy(); out['aday_n'] = j['size'].to_numpy()
    from prc_prefix import prefix_of
    apre = prefix_of(a.FLIGHT_mvt); dpre = prefix_of(d.FLIGHT_mvt)
    g2 = pd.DataFrame(dict(k=K(aapt[a_det], apre[a_det]), day=aday[a_det], st=a_st[a_det])).groupby(['k', 'day']).st.agg(['mean', 'size'])
    j2 = pd.DataFrame(dict(k=K(dapt, dpre), day=dday)).merge(g2.reset_index(), on=['k', 'day'], how='left')
    out['opday_stamp'] = j2['mean'].to_numpy(); out['opday_n'] = j2['size'].to_numpy()
    # 3. same-callsign INBOUND arrival? (arrival with same callsign-number is rare; skip) -> NM delay
    out['nm_delay'] = A3 - S
    return out, dapt, M, a_st, a_det, aapt, L
import sys; sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
open(f'{D}/prc_prefix.py', 'w').write('from prc.p_features import prefix_of\n')
res = {}
for which in ['training', 'ranking']:
    f, dapt, M, a_st, a_det, aapt, L = feats(which)
    f.to_parquet(f'{D}/stampfeat_{which}.parquet')
    mon = pd.to_datetime(M, unit='s').month
    amon = pd.to_datetime(L, unit='s').month
    res[which] = (f, dapt, mon, a_st, a_det, aapt, amon)
    print(which, 'coverage pa_stamped %.3f aday %.3f opday %.3f' % (f.pa_stamped.notna().mean(), f.aday_stamp.notna().mean(), f.opday_stamp.notna().mean()), 'peak', peak())
print('\n=== DRIFT: arrival in-block stamped at SIBT (detectable arrivals), % by airport and month')
rows = []
for which, (f, dapt, mon, a_st, a_det, aapt, amon) in res.items():
    for mm in [1, 7]:
        k = a_det & (amon == mm)
        s = pd.Series(a_st[k]).groupby(aapt[k]).mean() * 100
        rows.append(s.rename(f'{"2025" if which=="training" else "2026"}-{mm:02d}'))
print(pd.concat(rows, axis=1).round(2).to_string())
print('\n=== DRIFT: departure route-D share (NM record missing), % by airport and month (observable)')
rows = []
for which in ['training', 'ranking']:
    d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt'], which)
    mon = pd.to_datetime(d.MVT_TIME_UTC_mvt.to_numpy(), unit='s').month
    rd = d.AOBT_3_flt.isna().to_numpy(); P = rd & ((d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).to_numpy() > 7200)
    ap = S_(d.APT)
    for mm in [1, 7]:
        k = mon == mm
        rows.append((pd.Series(rd[k]).groupby(ap[k]).mean() * 100).rename(f'rd {"2025" if which=="training" else "2026"}-{mm:02d}'))
        rows.append((pd.Series(P[k]).groupby(ap[k]).sum()).rename(f'nP {"2025" if which=="training" else "2026"}-{mm:02d}'))
print(pd.concat(rows, axis=1).round(2).to_string())
print('peak', peak())
