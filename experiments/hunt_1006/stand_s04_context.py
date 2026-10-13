"""Label-free stand context per departure (2025 training and 2026 ranking, identical code):
  a_m   = in-block of the latest arrival at the same (APT, STAND) before our take-off T
  a_m2  = the one before it
  m_link= a_m has the same AIRCRAFT_TYPE and operator as us
  a_lk  = in-block of the latest same-type-and-operator arrival at the stand before T; n_after_lk = arrivals at S in (a_lk, T)
  dep_between = departures at S whose take-off lies in (a_m, T) (excluding us)
  nxt_dT = take-off of the next departure at S after T minus T; nxt_link = that departure has the type/operator of a_m
  prv_dT = T minus take-off of the previous departure at S
Writes cache/hunt/stand_ctx_{training,ranking}.parquet in the load('...', phase='DEP') row order."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'

for which in ('training', 'ranking'):
    t = pd.read_parquet(H + f'stand_ev_{which}.parquet', columns=['id', 'PHASE_mvt', 'APT', 'STAND_mvt', 'mvt', 'blk', 'sch', 'lobt', 'aobt3', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt'])
    t['key'] = (t.APT.astype(str) + '|' + t.STAND_mvt.astype(str)).astype('category')
    t['tk'] = (t.AIRCRAFT_TYPE_mvt.astype(str) + '|' + t.AIRCRAFT_OPERATOR_flt.astype(str))
    t['tkc'] = t.tk.astype('category').cat.codes.astype(np.int64)
    t['ki'] = t.key.cat.codes.astype(np.int64)
    dep = t[t.PHASE_mvt == 'DEP'].reset_index(drop=True)
    arr = t[(t.PHASE_mvt == 'ARR') & np.isfinite(t.blk) & t.STAND_mvt.notna()].reset_index(drop=True)
    n = len(dep)
    res = {c: np.full(n, np.nan) for c in ['a_m', 'a_m2', 'a_lk', 'n_after_lk', 'dep_between', 'nxt_dT', 'prv_dT', 'n_arr_sched_T']}
    res['m_link'] = np.zeros(n, bool); res['nxt_link'] = np.zeros(n, bool); res['a_m_tk'] = np.full(n, -1, np.int64)
    ag = {k: g for k, g in arr.groupby('ki', sort=False)}
    dep_valid = dep.STAND_mvt.notna().to_numpy()
    for k, gd in dep[dep_valid].groupby('ki', sort=False):
        idx = gd.index.to_numpy()
        T = gd.mvt.to_numpy(); tk = gd.tkc.to_numpy(); S = gd.sch.to_numpy()
        o = np.argsort(T, kind='stable'); Ts = T[o]
        # departures neighbours by take-off
        pos = np.searchsorted(Ts, T, side='right') - 1  # our index in sorted (ties: last)
        nxt = np.where(pos + 1 < len(Ts), Ts[np.minimum(pos + 1, len(Ts) - 1)], np.nan)
        res['nxt_dT'][idx] = nxt - T
        prv_pos = np.searchsorted(Ts, T, side='left') - 1
        res['prv_dT'][idx] = np.where(prv_pos >= 0, T - Ts[np.maximum(prv_pos, 0)], np.nan)
        nxt_tk = np.where(pos + 1 < len(Ts), tk[o][np.minimum(pos + 1, len(Ts) - 1)], -2)
        g = ag.get(k)
        if g is None:
            continue
        g = g.sort_values('blk'); a = g.blk.to_numpy(); atk = g.tkc.to_numpy()
        j = np.searchsorted(a, T, side='left')  # arrivals a[:j] strictly before T
        has = j > 0
        res['a_m'][idx[has]] = a[j[has] - 1]
        res['a_m_tk'][idx[has]] = atk[j[has] - 1]
        res['m_link'][idx[has]] = atk[j[has] - 1] == tk[has]
        has2 = j > 1
        res['a_m2'][idx[has2]] = a[j[has2] - 2]
        res['nxt_link'][idx[has]] = nxt_tk[has] == atk[j[has] - 1]
        res['n_arr_sched_T'][idx] = j - np.searchsorted(a, S, side='right')
        # departures with take-off in (a_m, T)
        am = np.where(has, a[np.maximum(j - 1, 0)], -np.inf)
        res['dep_between'][idx] = np.searchsorted(Ts, T, side='left') - np.searchsorted(Ts, am, side='right')
    # latest same-type-operator arrival before T (merge_asof), and how many stand arrivals followed it before T
    ar2 = arr[['ki', 'tkc', 'blk']].sort_values(['ki', 'blk']).reset_index(drop=True)
    ar2['rank'] = ar2.groupby('ki').cumcount()
    ar2['blk_lk'] = ar2.blk
    dv = dep.loc[dep_valid, ['ki', 'tkc', 'mvt']].reset_index()
    m = pd.merge_asof(dv.sort_values('mvt'), ar2.sort_values('blk')[['ki', 'tkc', 'blk', 'blk_lk', 'rank']],
                      left_on='mvt', right_on='blk', by=['ki', 'tkc'], direction='backward', allow_exact_matches=False)
    m = m.set_index('index').sort_index()
    res['a_lk'][m.index.to_numpy()] = m.blk_lk.to_numpy()
    # stand arrivals before T: count = j ; rank of a_lk within stand list
    cnt = pd.merge_asof(dv.sort_values('mvt'), ar2.sort_values('blk')[['ki', 'blk', 'rank']].rename(columns={'rank': 'rk_m'}),
                        left_on='mvt', right_on='blk', by='ki', direction='backward', allow_exact_matches=False).set_index('index').sort_index()
    res['n_after_lk'][m.index.to_numpy()] = (cnt.rk_m - m['rank']).to_numpy()
    out = pd.DataFrame({'id': dep.id.to_numpy(), 'APT': dep.APT.astype(str).to_numpy(), 'T': dep.mvt.to_numpy(),
                        'blk': dep.blk.to_numpy(), 'sch': dep.sch.to_numpy(), 'lobt': dep.lobt.to_numpy(),
                        'aobt3': dep.aobt3.to_numpy(), 'tkc': dep.tkc.to_numpy(), 'ki': dep.ki.to_numpy(), **res})
    out.to_parquet(H + f'stand_ctx_{which}.parquet')
    print(which, n, 'a_m present', np.isfinite(out.a_m).mean(), 'link present', np.isfinite(out.a_lk).mean())
