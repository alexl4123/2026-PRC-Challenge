"""routed_0926 step 0: rebuild the pieces the route-D corrector question needs and prove they reproduce
what is shipped. NO candidate is scored here (the rule is pre-registered in r01_cv.py / r02_2026.py).

2025 (12-fold LOMO OOF, DEP order, same rows as cache/v11/base.npz):
  corr_old = clip(m11d_oof_trailD, 0) - v9s_oof      the v10 corrector exactly as v17_anatomy carries it
  blendD   = adsb2 7-member blend, weights cache/w2/adsb2_weights.json 'route D'   (as v17_anatomy.py)
  base     = v17 structure WITHOUT the corrector:
               route D outside P  -> blendD
               zone P             -> v17s_oof - corr_old   (v17's P branch; the corrector is additive there)
               A/B/C              -> v17s_oof (never touched)
  R        = rows the corrector acts on (m11d): D & (~P | exp62 spec p < 0.1)
  CHECK    v17s_oof[D & ~P] == clip(blendD + corr_old, 0)   (float32 tolerance)
2026 ranking (DEP order = cache/final/lgb_components ids):
  corr_old_rk = clip(m11f trailD_full, 0) - m11f v9  (what exp69 added; carried additively to v17)
  blendD_rk   = exp77::blend(adsb2_final, adsb2 weights 'route D') with the production tail branch per LGB seed
  base_rk     = blendD_rk on route D outside P; v17 - corr_old_rk inside P
  R_rk        = D & (~P | spec_x_ranking p_spec < 0.1)   (m11d's ranking rule)
  CHECK       v17 file[D & ~P] ~= blendD_rk + corr_old_rk  (rounding over v10..v17, plus a known chain artefact on
              ~45 rows where v11 hit the 0 clip; see the check below)
Outputs: cache/routed_0926/arrays_2025.npz, arrays_2026.npz

    python experiments/routed_0926/r00_arrays.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'w2'))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import peak_rss_gb

OUT = CACHE / 'routed_0926'; OUT.mkdir(exist_ok=True)
FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3')


def fail(msg):
    sys.exit('FATAL: ' + msg)


def build_2025():
    from w2_eval import oof
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    W = json.load(open(CACHE / 'w2' / 'adsb2_weights.json'))
    if W['seeds'] != [0, 1, 2]:
        fail('adsb2 weights are not the 3-seed weights')
    wD = W['weights']['route D']
    blendD = sum(wD[f] * oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS if wD.get(f, 0) > 0)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    corr = v10 - v9
    v17 = np.load(CACHE / 'w2' / 'v17s_oof.npy').astype(float)
    n = len(y); DnP = D & ~P
    rec = np.clip(blendD + corr, 0, None)
    err = np.abs(rec[DnP] - v17[DnP]).max()
    print(f'2025 CHECK route D outside P: max |clip(blendD + corr_old) - v17s_oof| = {err:.4f} s over {DnP.sum():,} rows')
    if err > 0.01 * max(1.0, np.abs(v17[DnP]).max() / 1e4):
        fail('v17s_oof is not blendD + corrector on route D outside P')
    base = v17.copy(); base[DnP] = blendD[DnP]; base[P] = v17[P] - corr[P]
    sp = np.load(CACHE / 'exp62_spec_P.npy')[1]; pP = np.zeros(n); pP[np.where(P)[0]] = sp
    R = D & (~P | (pP < 0.1))
    outside = ~R & (np.abs(corr) > 1e-6)
    print(f'2025 corrector nonzero outside R: {int(outside.sum())} rows (must be 0)')
    if outside.any():
        fail('the corrector moves rows outside its m11d mask')
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'MVT_TIME_UTC_mvt', 'off_sched', 'y'])
    if not np.allclose(dd.y.to_numpy(), y):
        fail('dep_training order differs from base.npz')
    rows = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc', 'month'])
    np.savez(OUT / 'arrays_2025.npz', y=y, month=month, P=P, D=D, R=R, pP=pP, base=base, corr_old=corr, v17=v17,
             apt=dd.APT.astype(str).to_numpy(), hour=dd.MVT_TIME_UTC_mvt.dt.hour.to_numpy().astype(np.int16),
             off_sched=dd.off_sched.to_numpy(float), date_loc=rows.date_loc.to_numpy())
    print(f'2025: rows {n:,}; D {D.sum():,}  P {P.sum():,}  D outside P {DnP.sum():,}; corrector rows R {R.sum():,} '
          f'(outside P {(R & ~P).sum():,}, inside P {(R & P).sum():,})')
    for nm, k in (('D outside P', DnP), ('P & R', P & R)):
        print(f'  corr_old on {nm}: mean {corr[k].mean():+.1f}  sd {corr[k].std():.1f}  |c|>60: {int((np.abs(corr[k]) > 60).sum())}')


def build_2026():
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pt, et, in_P = c['ids'], c['pt'], c['et'], c['in_P'].astype(bool)
    rk = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet',
                         columns=[ID, 'APT', 'month', 'MVT_TIME_UTC_mvt', 'off_aobt3', 'off_sched'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('dep_ranking order differs from lgb_components')
    D = rk.off_aobt3.isna().to_numpy(); DnP = D & ~in_P
    if not np.array_equal(in_P, D & (rk.off_sched.to_numpy() > 7200)):
        fail('in_P != D & off_sched > 7200')
    F = CACHE / 'adsb2_final'
    if not np.array_equal(np.load(F / 'ranking_ids.npy', allow_pickle=True), ids):
        fail('adsb2_final in a different order')
    W = json.load(open(CACHE / 'w2' / 'adsb2_weights.json'))['weights']['route D']
    mem = {f: np.mean([np.load(F / f'{f}__seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)
           for f in FAMS if W.get(f, 0) > 0}
    out = []
    for s in range(pt.shape[0]):
        p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
        pbm = sum(W[f] * mem[f] for f in BULK if W.get(f, 0.0) > 0)
        o = Wb * p * et + (1 - p) * pbm
        if W.get('routed', 0.0) > 0:
            o = o + W['routed'] * mem['routed']
        out.append(np.clip(o, 0, None))
    blendD = np.mean(out, axis=0)
    z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
    if not np.array_equal(z['ids'], ids):
        fail('m11f candidates in a different order')
    corr = np.clip(z['trailD_full'], 0, None) - z['v9']
    v17 = pd.read_parquet(OUTPUTS / 'understanding-accordion_v17.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy(float)
    d = v17[DnP] - np.clip(blendD[DnP] + corr[DnP], 0, None)
    print(f'2026 CHECK route D outside P ({DnP.sum():,} rows): v17 - (blendD + corr_old): max |.| {np.abs(d).max():.2f} s, '
          f'mean {d.mean():+.3f}, rows |.|>2 s: {int((np.abs(d) > 2).sum())}; SSE of the mismatch {(d ** 2).sum():.3e} '
          f'vs SSE of the corrector there {(corr[DnP] ** 2).sum():.3e}')
    # Known delta-chain artefact (measured 09-26): where v11's stack correction drove v11 to the 0 clip, v15's delta
    # subtracted the UNclipped stack term, so ~45 rows (LSZH near 0, 4 LIRF rows up to +245 s) are not blend + corrector.
    # Tolerate it (its SSE is ~0.04% of the corrector's); fail on anything larger.
    if (np.abs(d) > 2).sum() > 100 or (d ** 2).sum() > 0.01 * (corr[DnP] ** 2).sum():
        fail('v17 on route D outside P is not blendD + corrector beyond the known chain artefact')
    bad = DnP.copy(); bad[DnP] = np.abs(d) > 2
    print('  mismatch rows by airport: ' + ', '.join(f'{a} {n}' for a, n in pd.Series(rk.APT.astype(str).to_numpy()[bad]).value_counts().items()))
    sx = np.load(CACHE / 'final' / 'spec_x_ranking.npz', allow_pickle=True)
    pos = pd.Series(np.arange(len(sx['ids'])), index=sx['ids'])
    pr = sx['p_spec'][pos.reindex(ids).to_numpy()]; pr = np.where(np.isfinite(pr), pr, 0.0)
    R = D & (~in_P | (pr < 0.1))
    outside = ~R & (np.abs(corr) > 1e-6)
    print(f'2026 corrector nonzero outside R: {int(outside.sum())} rows (must be 0)')
    if outside.any():
        fail('the 2026 corrector moves rows outside its m11d mask')
    base = v17.copy(); base[DnP] = blendD[DnP]; base[in_P] = v17[in_P] - corr[in_P]
    np.savez(OUT / 'arrays_2026.npz', ids=ids, month=rk.month.to_numpy(), P=in_P, D=D, R=R, pP=pr, base=base,
             corr_old=corr, v17=v17, chain_bad=bad, apt=rk.APT.astype(str).to_numpy(),
             hour=rk.MVT_TIME_UTC_mvt.dt.hour.to_numpy().astype(np.int16), off_sched=rk.off_sched.to_numpy(float))
    print(f'2026: rows {len(ids):,}; D {D.sum():,}  P {in_P.sum():,}  D outside P {DnP.sum():,}; corrector rows R {R.sum():,} '
          f'(outside P {(R & ~in_P).sum():,}, inside P {(R & in_P).sum():,})')


def main():
    t0 = time.time()
    build_2025()
    build_2026()
    print(f'peak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
