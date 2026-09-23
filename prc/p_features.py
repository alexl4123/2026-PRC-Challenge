"""Per-flight features that separate ARTEFACT from REAL inside P (exp62).

P = {AOBT_3_flt missing} AND {off_sched > 7200}. Inside P the whole _flt block is
NaN (training AND ranking), so the operator comes from the FLIGHT_mvt callsign
prefix and everything else from the movement tables:

  * rotation timing at our stand / pier from ARRIVAL in-blocks (arrival
    BLOCK_TIME, which ranking.parquet keeps) and other departures' takeoffs;
  * inbound existence (arrivals at our stand in the last 3/6/12 h);
  * schedule structure, callsign form, same-callsign records, operator activity;
  * our own flight's arrival record at its destination, when that is one of
    the ten airports (its _flt block can survive there);
  * in-fold target encodings (operator, stand, pier, destination; and the
    operator's stamping rate over all route-D departures with off_sched > 1800).

LEAKAGE GUARD: `load_set` blanks departure BLOCK_TIME on every DEP row before
any feature is computed. Measured in exp62 step 1 (claude_log 2026-09-23 §19).
"""
import numpy as np
import pandas as pd

from prc.config import TARGET
from prc.features import NEEDED
from prc.io import load
from prc.models.tail_specialist import population

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
H = 3600.0
W_TE = 8.0          # Beta prior weight, same as the specialist's table

LOADCOLS = sorted(set(NEEDED + [
    'MVT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt',
    'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']))



def secs(s):
    return (s - EPOCH).dt.total_seconds().to_numpy(dtype=float)


def pier_of(stand):
    # identical to prc.features.build: astype(str) first, so a missing stand is 'nan'
    return (stand.astype(str).str.extract(r'^([A-Za-z]+|\d)', expand=False)
            .fillna('NA').to_numpy(object))


def prefix_of(cs):
    s = cs.astype('object').where(cs.notna(), '')
    s = pd.Series(s.to_numpy(object)).astype(str)
    p3 = s.str[:3]
    pre = np.where(p3.str.isalpha() & (s.str.len() >= 3), p3, s.str[:2])
    pre = np.where(s.eq(''), 'NA', pre)
    return pre.astype(object)


class TIndex:
    """Per-key sorted event times, for searchsorted lookups."""

    def __init__(self, keys, t, **payload):
        df = pd.DataFrame({'k': pd.Series(keys, dtype=object), 't': t,
                           **{c: v for c, v in payload.items()}})
        df = df[df.k.notna() & np.isfinite(df.t)].sort_values(['k', 't'], kind='mergesort')
        df = df.reset_index(drop=True)
        self.t = df.t.to_numpy(float)
        self.pay = {c: df[c].to_numpy() for c in payload}
        self.idx = df.groupby('k', sort=False).indices

    def get(self, k):
        pos = self.idx.get(k) if k is not None else None
        if pos is None:
            return None, None
        return pos, self.t[pos]


def count_in(tt, lo, hi):
    """Number of times in [lo, hi)."""
    return int(np.searchsorted(tt, hi, 'left') - np.searchsorted(tt, lo, 'left'))


def load_set(which):
    d = load(which, columns=LOADCOLS + ([TARGET] if which == 'training' else []))
    dep = d.PHASE_mvt.eq('DEP').to_numpy()
    y = d.loc[dep, TARGET].astype(float).to_numpy() if which == 'training' else None
    # ---- LEAKAGE GUARD: departure BLOCK_TIME is the target; destroy it here ----
    d['BLOCK_TIME_UTC_mvt'] = d['BLOCK_TIME_UTC_mvt'].where(~dep)
    if TARGET in d.columns:
        d = d.drop(columns=TARGET)
    assert d.loc[dep, 'BLOCK_TIME_UTC_mvt'].isna().all()
    return d, dep, y


def candidates(d, dep_mask):
    """Candidate features for every P row of this set. Index = position among DEP rows."""
    dep = d[dep_mask].reset_index(drop=True)
    arr = d[~dep_mask].reset_index(drop=True)
    M, S = secs(dep.MVT_TIME_UTC_mvt), secs(dep.SCHED_TIME_UTC_mvt)
    os_ = M - S
    miss = dep.AOBT_3_flt.isna().to_numpy()
    P = population(os_, miss)
    Pidx = np.where(P)[0]
    apt = dep.APT.astype(str).to_numpy(object)
    stand = dep.STAND_mvt.astype('object').to_numpy(object)
    pier = pier_of(dep.STAND_mvt)
    cs = dep.FLIGHT_mvt.astype('object').to_numpy(object)
    pre = prefix_of(dep.FLIGHT_mvt)
    actype = dep.AIRCRAFT_TYPE_mvt.astype('object').to_numpy(object)
    ades = dep.ADES_mvt.astype('object').to_numpy(object)

    a_apt = arr.APT.astype(str).to_numpy(object)
    a_stand = arr.STAND_mvt.astype('object').to_numpy(object)
    a_pier = pier_of(arr.STAND_mvt)
    a_in, a_sched, a_land = (secs(arr.BLOCK_TIME_UTC_mvt), secs(arr.SCHED_TIME_UTC_mvt),
                             secs(arr.MVT_TIME_UTC_mvt))
    a_pre = prefix_of(arr.FLIGHT_mvt)
    a_cs = arr.FLIGHT_mvt.astype('object').to_numpy(object)
    a_type = arr.AIRCRAFT_TYPE_mvt.astype('object').to_numpy(object)
    a_adep = arr.ADEP_mvt.astype('object').to_numpy(object)
    a_aobt, a_lobt, a_eobt = (secs(arr.AOBT_3_flt), secs(arr.LOBT_flt), secs(arr.EOBT_1_flt))

    def skey(a, s):
        return np.array([f'{x}|{y}' if isinstance(y, str) else None for x, y in zip(a, s)],
                        dtype=object)

    # arrivals by stand / pier (in-block time), departures by stand (takeoff)
    ix_arr_stand = TIndex(skey(a_apt, a_stand), a_in, sched=a_sched, typ=a_type, pre=a_pre)
    ix_arr_pier = TIndex(np.array([f'{x}|{y}' for x, y in zip(a_apt, a_pier)], object), a_in)
    ix_dep_stand = TIndex(skey(apt, stand), M)
    # same callsign, same airport, other departure records (by SCHED; payload takeoff)
    ix_dep_cs = TIndex(skey(apt, cs), S, mvt=M)
    # operator prefix at the airport: arrivals and departures by SCHED, with route-D flags
    ix_arr_op = TIndex(np.array([f'{x}|{y}' for x, y in zip(a_apt, a_pre)], object), a_sched,
                       rd=np.isnan(a_aobt).astype(float))
    ix_dep_op = TIndex(np.array([f'{x}|{y}' for x, y in zip(apt, pre)], object), S,
                       rd=miss.astype(float))
    # our own flight's arrival record at its destination (ADEP|callsign|ADES)
    dkey = np.array([f'{x}|{c}|{z}' if isinstance(c, str) else None
                     for x, c, z in zip(a_adep, a_cs, a_apt)], object)
    ix_dst = TIndex(dkey, a_land, aobt=a_aobt, lobt=a_lobt, eobt=a_eobt, inb=a_in,
                    sched=a_sched)
    airports = set(np.unique(a_apt))

    # MVT_ID rank residual within (airport, month) departures
    mon = dep.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    g = pd.DataFrame({'apt': apt, 'mon': mon, 'id': dep.MVT_ID_mvt.to_numpy(float), 'S': S})
    id_resid = (g.groupby(['apt', 'mon']).id.rank(pct=True)
                - g.groupby(['apt', 'mon']).S.rank(pct=True)).to_numpy()

    nan = np.nan
    rows = []
    for i in Pidx:
        m_, s_ = M[i], S[i]
        r = {}
        # ---- rotation at our stand ----
        k = f'{apt[i]}|{stand[i]}' if isinstance(stand[i], str) else None
        pos, tt = ix_arr_stand.get(k)
        if tt is None:
            r.update(arr_prev_to_takeoff=nan, arr_prev_to_sched=nan, arr_prev_delay=nan,
                     arr_prev_same_type=nan, arr_prev_same_op=nan, arr_lastbefore_sched_gap=nan,
                     n_arr_sched_to_takeoff=nan, first_arr_after_sched_minus_takeoff=nan,
                     n_arr_3h=nan, n_arr_6h=nan, n_arr_12h=nan)
        else:
            j = np.searchsorted(tt, m_, 'left') - 1
            if j >= 0 and m_ - tt[j] < 48 * H:
                q = pos[j]
                r.update(arr_prev_to_takeoff=m_ - tt[j], arr_prev_to_sched=s_ - tt[j],
                         arr_prev_delay=tt[j] - ix_arr_stand.pay['sched'][q],
                         arr_prev_same_type=float(ix_arr_stand.pay['typ'][q] == actype[i]),
                         arr_prev_same_op=float(ix_arr_stand.pay['pre'][q] == pre[i]))
            else:
                r.update(arr_prev_to_takeoff=nan, arr_prev_to_sched=nan, arr_prev_delay=nan,
                         arr_prev_same_type=nan, arr_prev_same_op=nan)
            j2 = np.searchsorted(tt, s_, 'left') - 1
            r['arr_lastbefore_sched_gap'] = (s_ - tt[j2]) if (j2 >= 0 and s_ - tt[j2] < 48 * H) else nan
            r['n_arr_sched_to_takeoff'] = count_in(tt, s_, m_)
            nxt = j2 + 1
            r['first_arr_after_sched_minus_takeoff'] = (tt[nxt] - m_) if (
                nxt < len(tt) and tt[nxt] - s_ < 48 * H) else nan
            for h in (3, 6, 12):
                r[f'n_arr_{h}h'] = count_in(tt, m_ - h * H, m_)
        # other departures at our stand
        pos, tt = ix_dep_stand.get(k)
        if tt is None:
            r.update(n_dep_stand_between=nan, prev_dep_stand_to_takeoff=nan)
        else:
            r['n_dep_stand_between'] = count_in(tt, s_ + 1, m_ - 600)
            j = np.searchsorted(tt, m_, 'left') - 1
            r['prev_dep_stand_to_takeoff'] = (m_ - tt[j]) if (j >= 0 and m_ - tt[j] < 48 * H) else nan
        # ---- pier ----
        pos, tt = ix_arr_pier.get(f'{apt[i]}|{pier[i]}')
        if tt is None:
            r.update(pier_arr_prev_to_takeoff=nan, pier_arr_rate_sched_to_takeoff=nan)
        else:
            j = np.searchsorted(tt, m_, 'left') - 1
            r['pier_arr_prev_to_takeoff'] = (m_ - tt[j]) if (j >= 0 and m_ - tt[j] < 48 * H) else nan
            r['pier_arr_rate_sched_to_takeoff'] = count_in(tt, s_, m_) / ((m_ - s_) / H)
        # ---- same callsign, other departure records at this airport ----
        k2 = f'{apt[i]}|{cs[i]}' if isinstance(cs[i], str) else None
        pos, tt = ix_dep_cs.get(k2)
        if tt is None:
            r.update(samecs_prev_sched_gap=nan, samecs_next_sched_gap=nan,
                     samecs_takeoff_between=nan)
        else:
            mv = ix_dep_cs.pay['mvt'][pos]
            other = ~((tt == s_) & (mv == m_))
            tto, mvo = tt[other], mv[other]
            before, after = tto[tto < s_], tto[tto > s_]
            r['samecs_prev_sched_gap'] = (s_ - before.max()) / H if before.size and s_ - before.max() < 240 * H else nan
            r['samecs_next_sched_gap'] = (after.min() - s_) / H if after.size and after.min() - s_ < 240 * H else nan
            r['samecs_takeoff_between'] = float(((mvo > s_) & (mvo < m_)).sum())
        # ---- operator activity around our schedule ----
        kop = f'{apt[i]}|{pre[i]}'
        pos, tt = ix_arr_op.get(kop)
        if tt is None or pre[i] == 'NA':
            r.update(op_arr_n_6h=nan, op_arr_rd_share_6h=nan)
        else:
            lo, hi = np.searchsorted(tt, s_ - 3 * H), np.searchsorted(tt, s_ + 3 * H)
            r['op_arr_n_6h'] = float(hi - lo)
            r['op_arr_rd_share_6h'] = float(ix_arr_op.pay['rd'][pos][lo:hi].mean()) if hi > lo else nan
        pos, tt = ix_dep_op.get(kop)
        if tt is None or pre[i] == 'NA':
            r['op_dep_rd_share_6h'] = nan
        else:
            lo, hi = np.searchsorted(tt, s_ - 3 * H), np.searchsorted(tt, s_ + 3 * H)
            rd = ix_dep_op.pay['rd'][pos][lo:hi]
            n = hi - lo - 1                                     # exclude ourselves (route D)
            r['op_dep_rd_share_6h'] = float((rd.sum() - 1) / n) if n > 0 else nan
        # ---- our own flight's arrival record at its destination ----
        if isinstance(cs[i], str) and ades[i] in airports:
            pos, tt = ix_dst.get(f'{apt[i]}|{cs[i]}|{ades[i]}')
            q = None
            if tt is not None:
                j = np.searchsorted(tt, m_, 'right')
                if j < len(tt) and tt[j] - m_ < 20 * H:
                    q = pos[j]
            r['dst_found'] = float(q is not None)
            if q is not None:
                ao = ix_dst.pay['aobt'][q]
                r['dst_has_aobt'] = float(np.isfinite(ao))
                r['dst_aobt_to_takeoff'] = m_ - ao
                r['dst_aobt_minus_sched'] = ao - s_
                r['dst_lobt_to_takeoff'] = m_ - ix_dst.pay['lobt'][q]
                r['dst_eobt_to_takeoff'] = m_ - ix_dst.pay['eobt'][q]
                r['dst_arr_delay'] = ix_dst.pay['inb'][q] - ix_dst.pay['sched'][q]
            else:
                r.update(dst_has_aobt=nan, dst_aobt_to_takeoff=nan, dst_aobt_minus_sched=nan,
                         dst_lobt_to_takeoff=nan, dst_eobt_to_takeoff=nan, dst_arr_delay=nan)
        else:
            r.update(dst_found=nan, dst_has_aobt=nan, dst_aobt_to_takeoff=nan,
                     dst_aobt_minus_sched=nan, dst_lobt_to_takeoff=nan,
                     dst_eobt_to_takeoff=nan, dst_arr_delay=nan)
        rows.append(r)

    out = pd.DataFrame(rows, index=Pidx)
    st = dep.SCHED_TIME_UTC_mvt.iloc[Pidx]
    out['sched_hour'] = st.dt.hour.to_numpy()
    out['sched_minute'] = st.dt.minute.to_numpy()
    out['sched_round_hour'] = ((st.dt.minute == 0) & (st.dt.second == 0)).astype(float).to_numpy()
    c = pd.Series(cs[Pidx], dtype=object)
    cstr = c.fillna('').astype(str)
    out['cs_missing'] = c.isna().astype(float).to_numpy()
    out['cs_len'] = np.where(c.isna(), np.nan, cstr.str.len())
    out['cs_garbled'] = np.where(c.isna(), np.nan,
                                 (cstr.str[:3].str.isalpha() & (cstr.str.len() >= 4)
                                  & cstr.str[3:4].str.isalpha()).astype(float))
    out['cs_suffix_alpha'] = np.where(c.isna(), np.nan, cstr.str[-1:].str.isalpha().astype(float))
    fr = dep.FLIGHT_RULE_mvt.astype('object').iloc[Pidx]
    out['flight_rule'] = fr.map({'I': 0.0, 'V': 1.0}).fillna(2.0).to_numpy()
    out['id_rank_resid'] = id_resid[Pidx]
    # keys for the in-fold target encodings
    out['k_op'] = [f'{a}|{p}' for a, p in zip(apt[Pidx], pre[Pidx])]
    out['k_stand'] = [f'{a}|{s}' for a, s in zip(apt[Pidx], stand[Pidx])]
    out['k_pier'] = [f'{a}|{p}' for a, p in zip(apt[Pidx], pier[Pidx])]
    out['k_ades'] = [f'{a}|{z}' for a, z in zip(apt[Pidx], ades[Pidx])]
    out['apt'] = apt[Pidx]
    out['off_sched'] = os_[Pidx]
    out['month'] = mon[Pidx]
    # route-D pool (off_sched > 1800) for the wider operator stamping-rate encoding
    rd_pool = miss & np.isfinite(os_) & (os_ > 1800)
    pool = pd.DataFrame({'k_op': [f'{a}|{p}' for a, p in zip(apt[rd_pool], pre[rd_pool])],
                         'apt': apt[rd_pool], 'month': mon[rd_pool],
                         'off_sched': os_[rd_pool], 'pos': np.where(rd_pool)[0]})
    return out, pool, P


# ---------------------------------------------------------------------------
# target encodings, in-fold
# ---------------------------------------------------------------------------
def te_fit(keys, t, apt, w=W_TE):
    g = float(t.mean()) if len(t) else 0.0
    df = pd.DataFrame({'k': keys, 't': t, 'a': apt})
    ga = df.groupby('a').t.agg(['sum', 'count'])
    arate = ((ga['sum'] + w * g) / (ga['count'] + w)).to_dict()
    gk = df.groupby('k').t.agg(['sum', 'count'])
    return g, arate, gk


def te_apply(fitted, keys, apt, w=W_TE):
    g, arate, gk = fitted
    base = np.array([arate.get(a, g) for a in apt], float)
    s = gk['sum'].reindex(keys).to_numpy(float)
    n = gk['count'].reindex(keys).to_numpy(float)
    seen = np.isfinite(n)
    out = base.copy()
    out[seen] = (s[seen] + w * base[seen]) / (n[seen] + w)
    return out, seen


FAMILIES = {
    'rot': ['arr_prev_to_takeoff', 'arr_prev_to_sched', 'arr_prev_delay', 'arr_prev_same_type',
            'arr_prev_same_op', 'arr_lastbefore_sched_gap', 'n_arr_sched_to_takeoff',
            'first_arr_after_sched_minus_takeoff', 'n_dep_stand_between',
            'prev_dep_stand_to_takeoff'],
    'pier': ['pier_arr_prev_to_takeoff', 'pier_arr_rate_sched_to_takeoff'],
    'inb': ['n_arr_3h', 'n_arr_6h', 'n_arr_12h'],
    'op': ['te_op', 'te_op_rd'],
    'te': ['te_op', 'te_op_rd', 'te_stand', 'te_pier', 'te_ades'],
    'sched': ['sched_hour', 'sched_minute', 'sched_round_hour'],
    'cs': ['cs_missing', 'cs_len', 'cs_garbled', 'cs_suffix_alpha', 'flight_rule'],
    'other': ['samecs_prev_sched_gap', 'samecs_next_sched_gap', 'samecs_takeoff_between',
              'op_arr_n_6h', 'op_arr_rd_share_6h', 'op_dep_rd_share_6h', 'id_rank_resid',
              'dst_found', 'dst_has_aobt', 'dst_aobt_to_takeoff', 'dst_aobt_minus_sched',
              'dst_lobt_to_takeoff', 'dst_eobt_to_takeoff', 'dst_arr_delay'],
}
FAMILIES['all'] = list(dict.fromkeys(sum(FAMILIES.values(), [])))
TE_COLS = {'te_op': 'k_op', 'te_stand': 'k_stand', 'te_pier': 'k_pier', 'te_ades': 'k_ades'}


def te_features(tr_cand, tr_tail, tr_pool, tr_pool_art, out_cand):
    """All TE columns for one fit.

    tr_cand / tr_tail     : P rows of the TRAINING months and their tail label
    tr_pool / tr_pool_art : route-D pool rows (off_sched > 1800) of the training
                            months and their artefact label
    out_cand              : rows to score (validation month, or ranking)
    Training rows are encoded by inner leave-one-month-out, so the GBM never sees
    a row's own label; out rows are encoded from all training rows. Identical to
    the step-1 screen's original in-fold encoder (OOF verified bit-identical).
    """
    te_tr = pd.DataFrame(index=tr_cand.index)
    te_out = pd.DataFrame(index=out_cand.index)
    mon_tr = tr_cand['month'].to_numpy()
    srcs = {c: (tr_cand[k].to_numpy(object), tr_cand['apt'].to_numpy(object), mon_tr,
                np.asarray(tr_tail, float), k) for c, k in TE_COLS.items()}
    srcs['te_op_rd'] = (tr_pool['k_op'].to_numpy(object), tr_pool['apt'].to_numpy(object),
                        tr_pool['month'].to_numpy(), np.asarray(tr_pool_art, float), 'k_op')
    for c, (sk, sa, sm, st, k) in srcs.items():
        te_out[c], _ = te_apply(te_fit(sk, st, sa), out_cand[k].to_numpy(object),
                                out_cand['apt'].to_numpy(object))
        col = np.full(len(tr_cand), np.nan)
        for m in np.unique(mon_tr):
            f = te_fit(sk[sm != m], st[sm != m], sa[sm != m])
            rows = mon_tr == m
            col[rows], _ = te_apply(f, tr_cand[k].to_numpy(object)[rows],
                                    tr_cand['apt'].to_numpy(object)[rows])
        te_tr[c] = col
    return te_tr, te_out
