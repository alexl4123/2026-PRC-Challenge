"""v11 step 2 -- features for the residual stacker, training (2025) and ranking (2026).

FAMILIES (all from ranking.parquet columns; arrivals keep their in-block time there):
  basic    apt, off_aobt3, off_sched, hour, aobt_missing
  clocks   nm_delay (AOBT_3-SCHED), off_iobt, lobt_m_iobt, eobt_m_lobt, a3_m_eobt, a3_m_lobt,
           eobt_m_sched, iobt_m_sched
  stamp    pa_stamped, pa_inb_m_sibt, pa_ldg_m_sibt  (previous ARRIVAL at our stand, in-block
           before our SCHED within 12 h: was its in-block stamped at its scheduled in-block?)
  ident    ades (categorical)
  keys     k_op, k_stand, k_ades  (codes; the nested target encodings are made in v11_stack.py)
  turnover BACKWARD si_before_sched, si_same_typ, sd_prev_to ;
           LOOK-AHEAD sl_to, sl_after_sched, sl_same_typ, sl_same_op, sn_minus_to, sn_same_op,
           sn_same_typ, sa_after_to, sd_next_to  ('isocap': each month from its own events only,
           every gap |g| > 48 h -> NaN together with its match flags)
Every event-based feature (stamp, turnover) is built MONTH BY MONTH from that month's events only,
in training and in ranking alike, so no ranking feature can cross between January and July.

CHECK 1: the departure frame handed to the builders has NO BLOCK_TIME / TAXITIME column (asserted).
Labels for training (y, stamping label) are written separately by `labels()` and are used only as
the stacker target and as the source of the nested target encodings.

Writes cache/v11/feat_{training,ranking}.parquet (DEP order of load(which, phase='DEP')) and
cache/v11/labels_training.npz; prints CHECK 2 (coverage) and CHECK 3 (isolation).
    python experiments/v11/v11_features.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from prc.io import load, peak_rss_gb
from prc.config import CACHE, ID

t0 = time.time()
OUT = CACHE / 'v11'; OUT.mkdir(exist_ok=True)
H = 3600.0
STR = ['APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']
TS_DEP = ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt']
TS_ARR = ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']
FORBIDDEN_DEP = ('BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt')
TURN = ['sl_to', 'sl_after_sched', 'sl_same_typ', 'sl_same_op', 'si_before_sched', 'si_same_typ',
        'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to', 'sd_next_to', 'sd_prev_to']


def secs(col):
    return pc.divide(pc.cast(pc.cast(col, pa.timestamp('us', 'UTC')), pa.int64()), 1e6).to_numpy(zero_copy_only=False).astype(float)


def read(which):
    """DEP frame WITHOUT block/taxi columns, and ARR frame (with arrival in-block)."""
    cols = [ID, 'PHASE_mvt'] + STR + sorted(set(TS_DEP + TS_ARR))
    tb = pq.read_table(CACHE / f'{which}.parquet', columns=cols, read_dictionary=STR + ['PHASE_mvt'])
    isdep = pc.equal(pc.cast(tb['PHASE_mvt'], pa.string()), 'DEP')
    out = {}
    for ph, mask, ts in (('DEP', isdep, TS_DEP), ('ARR', pc.invert(isdep), TS_ARR)):
        t = tb.filter(mask)
        f = pd.DataFrame({ID: t[ID].to_numpy()})
        for c in STR:
            f[c] = t[c].to_pandas().astype(object).where(lambda s: s.notna(), None).to_numpy()
        for c in ts:
            f[c] = secs(t[c])
        out[ph] = f
    dep, arr = out['DEP'], out['ARR']
    bad = [c for c in FORBIDDEN_DEP if c in dep.columns]
    assert not bad, f'CHECK 1 FAILED: departure frame carries {bad}'
    return dep, arr


def labels():
    """Training labels -- used ONLY for the stacker target and the nested target encodings."""
    tb = pq.read_table(CACHE / 'training.parquet', columns=['PHASE_mvt', 'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt',
                                                             'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt'],
                       read_dictionary=['PHASE_mvt'])
    tb = tb.filter(pc.equal(pc.cast(tb['PHASE_mvt'], pa.string()), 'DEP'))
    y = tb['TAXITIME_SEC_mvt'].to_numpy(zero_copy_only=False).astype(float)
    B, S, A3, M = (secs(tb[c]) for c in ['BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt'])
    stamp_mask = np.isfinite(A3) & (A3 - S > 600) & (M - A3 > 0)          # delayed route-A: stamping detectable
    stamp_lab = (np.abs(B - S) < 60).astype(float)
    return y, stamp_mask, stamp_lab


def s_(a):
    return np.array(['NA' if v is None else str(v) for v in a], dtype=object)


def month_isolated(dep, arr, dmon, amon):
    """stamp (pa_*) and turnover (isocap) features, each month from its own events only."""
    F = {c: np.full(len(dep), np.nan) for c in ['pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt'] + TURN}
    kd = s_(dep.APT) + '|' + s_(dep.STAND_mvt); ka = s_(arr.APT) + '|' + s_(arr.STAND_mvt)
    codes, _ = pd.factorize(np.concatenate([kd, ka])); kdc, kac = codes[:len(kd)], codes[len(kd):]
    tc, _ = pd.factorize(np.concatenate([s_(dep.AIRCRAFT_TYPE_mvt), s_(arr.AIRCRAFT_TYPE_mvt)])); tdc, tac = tc[:len(kd)], tc[len(kd):]
    oc, _ = pd.factorize(np.concatenate([pd.Series(s_(dep.FLIGHT_mvt)).str[:3].to_numpy(), pd.Series(s_(arr.FLIGHT_mvt)).str[:3].to_numpy()]))
    odc, oac = oc[:len(kd)], oc[len(kd):]
    dok = dep.STAND_mvt.notna().to_numpy() & (dep.STAND_mvt.to_numpy() != None)
    aok = arr.STAND_mvt.notna().to_numpy() & (arr.STAND_mvt.to_numpy() != None)
    t, ts = dep.MVT_TIME_UTC_mvt.to_numpy(), dep.SCHED_TIME_UTC_mvt.to_numpy()
    tb, sa, la = arr.BLOCK_TIME_UTC_mvt.to_numpy(), arr.SCHED_TIME_UTC_mvt.to_numpy(), arr.MVT_TIME_UTC_mvt.to_numpy()
    for m in np.unique(dmon):
        rows = dok & (dmon == m); ev = aok & (amon == m) & np.isfinite(tb); evd = dok & (dmon == m)
        q = pd.DataFrame(dict(k=kdc[rows], t=t[rows], ts=ts[rows], i=np.where(rows)[0], typ=tdc[rows], op=odc[rows]))
        e = pd.DataFrame(dict(k=kac[ev], tl=tb[ev], ltyp=tac[ev], lop=oac[ev], stp=(np.abs(tb[ev] - sa[ev]) < 60).astype(float),
                              inb=tb[ev] - sa[ev], ldg=la[ev] - sa[ev])).sort_values('tl')
        def put(mm, name, vals):
            F[name][mm.i.to_numpy()] = vals
        # stamp: previous arrival at our stand with in-block <= our SCHED, within 12 h
        mm = pd.merge_asof(q.sort_values('ts'), e, left_on='ts', right_on='tl', by='k', direction='backward', tolerance=12 * H)
        put(mm, 'pa_stamped', mm.stp.to_numpy()); put(mm, 'pa_inb_m_sibt', mm.inb.to_numpy()); put(mm, 'pa_ldg_m_sibt', mm.ldg.to_numpy())
        # turnover (outside-2 definitions, c4_stand.py)
        mm = pd.merge_asof(q.sort_values('t'), e, left_on='t', right_on='tl', by='k', direction='backward'); h = mm.tl.notna()
        put(mm, 'sl_to', (mm.t - mm.tl).to_numpy()); put(mm, 'sl_after_sched', (mm.tl - mm.ts).to_numpy())
        put(mm, 'sl_same_typ', np.where(h, (mm.ltyp == mm.typ).astype(float), np.nan)); put(mm, 'sl_same_op', np.where(h, (mm.lop == mm.op).astype(float), np.nan))
        mm = pd.merge_asof(q.sort_values('ts'), e, left_on='ts', right_on='tl', by='k', direction='backward'); h = mm.tl.notna()
        put(mm, 'si_before_sched', (mm.ts - mm.tl).to_numpy()); put(mm, 'si_same_typ', np.where(h, (mm.ltyp == mm.typ).astype(float), np.nan))
        mm = pd.merge_asof(q.sort_values('ts'), e, left_on='ts', right_on='tl', by='k', direction='forward', allow_exact_matches=False); h = mm.tl.notna()
        put(mm, 'sn_minus_to', (mm.tl - mm.t).to_numpy()); put(mm, 'sn_same_op', np.where(h, (mm.lop == mm.op).astype(float), np.nan))
        put(mm, 'sn_same_typ', np.where(h, (mm.ltyp == mm.typ).astype(float), np.nan))
        mm = pd.merge_asof(q.sort_values('t'), e, left_on='t', right_on='tl', by='k', direction='forward')
        put(mm, 'sa_after_to', (mm.tl - mm.t).to_numpy())
        dq = pd.DataFrame(dict(k=kdc[evd], td=t[evd])).sort_values('td')
        mm = pd.merge_asof(q.sort_values('t'), dq, left_on='t', right_on='td', by='k', direction='forward', allow_exact_matches=False)
        put(mm, 'sd_next_to', (mm.td - mm.t).to_numpy())
        mm = pd.merge_asof(q.sort_values('t'), dq, left_on='t', right_on='td', by='k', direction='backward', allow_exact_matches=False)
        put(mm, 'sd_prev_to', (mm.t - mm.td).to_numpy())
    lim = 48 * H
    for g, flags in [('sl_to', ['sl_same_typ', 'sl_same_op', 'sl_after_sched']), ('si_before_sched', ['si_same_typ']),
                     ('sn_minus_to', ['sn_same_op', 'sn_same_typ']), ('sa_after_to', []), ('sd_next_to', []), ('sd_prev_to', [])]:
        bad = np.abs(F[g]) > lim
        for c in [g] + flags:
            F[c][bad] = np.nan
    return F


def build(which):
    dep, arr = read(which)
    M, S, A3, E1, LO, IO = (dep[c].to_numpy() for c in TS_DEP)
    dmon = pd.to_datetime(M, unit='s').month.to_numpy(); amon = pd.to_datetime(arr.MVT_TIME_UTC_mvt.to_numpy(), unit='s').month.to_numpy()
    f = pd.DataFrame({ID: dep[ID].to_numpy(), 'month': dmon})
    f['apt_s'] = s_(dep.APT); f['ades_s'] = s_(dep.ADES_mvt)
    f['k_op'] = f.apt_s + '|' + s_(dep.AIRCRAFT_OPERATOR_flt); f['k_stand'] = f.apt_s + '|' + s_(dep.STAND_mvt); f['k_ades'] = f.apt_s + '|' + f.ades_s
    f['off_aobt3'] = M - A3; f['off_sched'] = M - S; f['hour'] = pd.to_datetime(M, unit='s').hour.to_numpy(); f['aobt_missing'] = np.isnan(A3).astype(float)
    f['nm_delay'] = A3 - S; f['off_iobt'] = M - IO; f['lobt_m_iobt'] = LO - IO; f['eobt_m_lobt'] = E1 - LO
    f['a3_m_eobt'] = A3 - E1; f['a3_m_lobt'] = A3 - LO; f['eobt_m_sched'] = E1 - S; f['iobt_m_sched'] = IO - S
    for c, v in month_isolated(dep, arr, dmon, amon).items():
        f[c] = v
    return f


tr = build('training'); print(f'training features {tr.shape} {time.time()-t0:.0f}s', flush=True)
rk = build('ranking'); print(f'ranking features {rk.shape} {time.time()-t0:.0f}s', flush=True)
# row order = load(which, phase='DEP')
for which, f in (('training', tr), ('ranking', rk)):
    ref = load(which, phase='DEP', columns=[ID])[ID].to_numpy()
    assert np.array_equal(ref, f[ID].to_numpy()), f'{which}: row order differs from load(phase=DEP)'
comp = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
assert np.array_equal(comp['ids'], rk[ID].to_numpy()), 'ranking order differs from cache/final/lgb_components.npz ids'
print('row order: training == load(DEP); ranking == load(DEP) == lgb_components ids')
# shared categorical vocabularies (from TRAINING); unseen in ranking -> NaN
for c in ['apt_s', 'ades_s']:
    voc = pd.Index(sorted(tr[c].unique()))
    tr[c.replace('_s', '')] = voc.get_indexer(tr[c]).astype(float)
    v = voc.get_indexer(rk[c]).astype(float); v[v < 0] = np.nan; rk[c.replace('_s', '')] = v
for c in ['k_op', 'k_stand', 'k_ades']:           # joint codes so nested encodings can be looked up for ranking
    codes, _ = pd.factorize(np.concatenate([tr[c].to_numpy(), rk[c].to_numpy()]))
    tr[c] = codes[:len(tr)].astype(np.int64); rk[c] = codes[len(tr):].astype(np.int64)
NUM = [c for c in tr.columns if c not in (ID, 'apt_s', 'ades_s', 'k_op', 'k_stand', 'k_ades', 'month')]
for f in (tr, rk):
    for c in NUM:
        f[c] = f[c].astype('float32')
tr.drop(columns=['apt_s', 'ades_s']).to_parquet(OUT / 'feat_training.parquet')
rk.drop(columns=['apt_s', 'ades_s']).to_parquet(OUT / 'feat_ranking.parquet')
y, smask, slab = labels()
np.savez(OUT / 'labels_training.npz', y=y, stamp_mask=smask, stamp_lab=slab)
print('CHECK 1: departure frames built without', FORBIDDEN_DEP, '(asserted in read()); labels written separately')

# CHECK 2 -- coverage, 2025 vs 2026, per scored month
print('\nCHECK 2 -- coverage (non-missing share), per scored month; FLAG = 2026 below 2025 by > 5 pp')
rows = []
for c in NUM + ['k_op', 'k_stand', 'k_ades']:
    r = {'feature': c}
    for nm, f in (('25', tr), ('26', rk)):
        for m in (1, 7):
            v = f.loc[f.month == m, c].to_numpy()
            r[f'{nm}-{m:02d}'] = float(np.mean(np.isfinite(v.astype(float)))) if c not in ('k_op', 'k_stand', 'k_ades') else np.nan
    rows.append(r)
cov = pd.DataFrame(rows).set_index('feature')
# key coverage = share of ranking keys seen in training (for the encodings)
for c in ['k_op', 'k_stand', 'k_ades']:
    seen = set(tr[c].to_numpy())
    for m in (1, 7):
        cov.loc[c, f'26-{m:02d}'] = np.isin(rk.loc[rk.month == m, c].to_numpy(), list(seen)).mean()
        cov.loc[c, f'25-{m:02d}'] = 1.0
cov['FLAG'] = np.where((cov['26-01'] < cov['25-01'] - 0.05) | (cov['26-07'] < cov['25-07'] - 0.05), 'FLAG', '')
print(cov.round(4).to_string())
cov.to_csv(OUT / 'coverage.csv')

# CHECK 3 -- month isolation in ranking
print('\nCHECK 3 -- month isolation')
gaps = ['sl_to', 'si_before_sched', 'sn_minus_to', 'sa_after_to', 'sd_next_to', 'sd_prev_to']
for nm, f in (('training', tr), ('ranking', rk)):
    mx = {g: float(np.nanmax(np.abs(f[g].to_numpy()))) / H for g in gaps}
    mxp = float(np.nanmax(np.abs(f['pa_inb_m_sibt'].to_numpy() - f['pa_ldg_m_sibt'].to_numpy())))
    print(f'  {nm}: max |turnover gap| (h) ' + ' '.join(f'{g} {v:.1f}' for g, v in mx.items()))
    assert all(v <= 48.0001 for v in mx.values()), f'{nm}: a turnover gap exceeds 48 h'
print('  every event feature is built per month from that month\'s events only (month_isolated); '
      'ranking months present:', sorted(rk.month.unique()), '; all turnover gaps <= 48 h (asserted); '
      'stamp lookback tolerance 12 h')
print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time()-t0:.0f}s')
