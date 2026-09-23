"""Rebuild outside2's stand-turnover family (c4_stand.py logic) with integer keys, in three constructions:
 cont   : all 2025 events (outside2's training construction; must equal c4_stand.parquet)
 pair   : month m rows see only events of months {m, m+6}  (mimics ranking.parquet = Jan+Jul in one file)
 isocap : month m rows see only month m events, and every gap |g| > 48 h -> NaN (with its match flags)"""
import time
from common import *
t0 = time.time()
dep = pd.read_parquet(f'{D}/training_DEP.parquet', columns=['APT', 'STAND_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
arr = pd.read_parquet(f'{D}/training_ARR.parquet', columns=['APT', 'STAND_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'])
kd = K(S_(dep.APT), S_(dep.STAND_mvt)); ka = K(S_(arr.APT), S_(arr.STAND_mvt))
codes, uniq = pd.factorize(np.concatenate([kd, ka])); kd_c, ka_c = codes[:len(kd)], codes[len(kd):]
tc, tu = pd.factorize(np.concatenate([S_(dep.AIRCRAFT_TYPE_mvt), S_(arr.AIRCRAFT_TYPE_mvt)])); td_c, ta_c = tc[:len(kd)], tc[len(kd):]
oc, ou = pd.factorize(np.concatenate([pd.Series(S_(dep.FLIGHT_mvt)).str[:3].to_numpy(), pd.Series(S_(arr.FLIGHT_mvt)).str[:3].to_numpy()])); od_c, oa_c = oc[:len(kd)], oc[len(kd):]
okd = dep.STAND_mvt.notna().to_numpy()
t = dep.MVT_TIME_UTC_mvt.to_numpy(); ts = dep.SCHED_TIME_UTC_mvt.to_numpy(); tb = arr.BLOCK_TIME_UTC_mvt.to_numpy()
mdep = pd.to_datetime(t, unit='s').month.to_numpy(); marr = pd.to_datetime(arr.MVT_TIME_UTC_mvt.to_numpy(), unit='s').month.to_numpy()
del dep, arr, kd, ka
COLS = ['sl_to', 'sl_after_sched', 'sl_same_typ', 'sl_same_op', 'si_before_sched', 'si_same_typ', 'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to', 'sd_next_to', 'sd_prev_to']
def build(rows, ev_arr, ev_dep):
    """features for dep rows `rows`, using arrival events ev_arr and departure events ev_dep (boolean masks)."""
    F = {c: np.full(len(t), np.nan, np.float32) for c in COLS}
    q = pd.DataFrame(dict(k=kd_c[rows], t=t[rows], ts=ts[rows], i=np.where(rows)[0], typ=td_c[rows], op=od_c[rows]))
    e = pd.DataFrame(dict(k=ka_c[ev_arr], tl=tb[ev_arr], ltyp=ta_c[ev_arr], lop=oa_c[ev_arr])).dropna(subset=['tl']).sort_values('tl')
    def put(m, name, vals): F[name][m.i.to_numpy()] = vals
    m = pd.merge_asof(q.sort_values('t'), e, left_on='t', right_on='tl', by='k', direction='backward')
    has = m.tl.notna()
    put(m, 'sl_to', (m.t - m.tl).to_numpy()); put(m, 'sl_after_sched', (m.tl - m.ts).to_numpy())
    put(m, 'sl_same_typ', np.where(has, (m.ltyp == m.typ).astype(float), np.nan)); put(m, 'sl_same_op', np.where(has, (m.lop == m.op).astype(float), np.nan))
    m = pd.merge_asof(q.sort_values('ts'), e, left_on='ts', right_on='tl', by='k', direction='backward'); has = m.tl.notna()
    put(m, 'si_before_sched', (m.ts - m.tl).to_numpy()); put(m, 'si_same_typ', np.where(has, (m.ltyp == m.typ).astype(float), np.nan))
    m = pd.merge_asof(q.sort_values('ts'), e, left_on='ts', right_on='tl', by='k', direction='forward', allow_exact_matches=False); has = m.tl.notna()
    put(m, 'sn_minus_to', (m.tl - m.t).to_numpy()); put(m, 'sn_same_op', np.where(has, (m.lop == m.op).astype(float), np.nan)); put(m, 'sn_same_typ', np.where(has, (m.ltyp == m.typ).astype(float), np.nan))
    m = pd.merge_asof(q.sort_values('t'), e, left_on='t', right_on='tl', by='k', direction='forward')
    put(m, 'sa_after_to', (m.tl - m.t).to_numpy())
    dq = pd.DataFrame(dict(k=kd_c[ev_dep], td=t[ev_dep])).sort_values('td')
    m = pd.merge_asof(q.sort_values('t'), dq, left_on='t', right_on='td', by='k', direction='forward', allow_exact_matches=False)
    put(m, 'sd_next_to', (m.td - m.t).to_numpy())
    m = pd.merge_asof(q.sort_values('t'), dq, left_on='t', right_on='td', by='k', direction='backward', allow_exact_matches=False)
    put(m, 'sd_prev_to', (m.t - m.td).to_numpy())
    return F
def cap(F, h=48):
    lim = h * 3600.0
    for g, flags in [('sl_to', ['sl_same_typ', 'sl_same_op', 'sl_after_sched']), ('si_before_sched', ['si_same_typ']),
                     ('sn_minus_to', ['sn_same_op', 'sn_same_typ']), ('sa_after_to', []), ('sd_next_to', []), ('sd_prev_to', [])]:
        bad = np.abs(F[g]) > lim
        for c in [g] + flags: F[c][bad] = np.nan
    return F
allA = np.ones(len(tb), bool); allD = okd
out = {}
F = build(okd, allA, allD); out['cont'] = F
ref = pd.read_parquet('/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/ideas_outside2/c4_stand.parquet')
for c in COLS:
    a, b = F[c], ref[c].to_numpy(np.float32)
    same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1)
    print(f'cont vs c4_stand {c:16s} agree {same.mean():.6f}')
for mode in ['pair', 'isocap']:
    G = {c: np.full(len(t), np.nan, np.float32) for c in COLS}
    for mm in range(1, 13):
        mset = [mm, (mm + 5) % 12 + 1] if mode == 'pair' else [mm]
        rows = okd & (mdep == mm)
        Fm = build(rows, np.isin(marr, mset), okd & np.isin(mdep, mset))
        for c in COLS: G[c][rows] = Fm[c][rows]
    if mode == 'isocap': G = cap(G)
    out[mode] = G
    print(mode, 'built', f'{time.time()-t0:.0f}s')
for mode, F in out.items():
    pd.DataFrame(F).to_parquet(f'{D}/rv_stand_{mode}.parquet')
big = 30 * 86400
mon = mdep
for mode in ['cont', 'pair', 'isocap']:
    F = out[mode]
    for mm in [1, 7]:
        k = mon == mm
        print(f'{mode:6s} m{mm}: rows with any |gap|>30d: {int(np.any([np.abs(F[c][k]) > big for c in ["sl_to","si_before_sched","sn_minus_to","sa_after_to","sd_next_to","sd_prev_to"]], axis=0).sum())}, '
              f'NaN sa_after_to {np.isnan(F["sa_after_to"][k]).mean():.4f}')
print('peak', peak(), f'{time.time()-t0:.0f}s')
