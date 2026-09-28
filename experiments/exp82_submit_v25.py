"""EXP82 -- v25 = v24 with Rome's zone-P tail probability from TabICL + the carrier stamping rate.

DECISION (user, 2026-09-28): a deliberate JUDGEMENT CALL. experiments/tail/t06_carrier_stamp.py arm (a) FAILED its
pre-registered rule (12m -1.52 [-3.29,-0.05], Jan+Jul -2.25 [-5.40,+0.02], but January +0.06 on ~24 Rome P rows and
the gain concentrated -- R3, R5); the descriptive checks (t07) showed a real, stable, transferable carrier signal
(H1/H2 2025 corr +0.99; gains in every carrier-sample-size bin; 2026 carriers better covered). Shipped as ONE live test.

Only the 181 Rome zone-P ranking rows change:
    v25 = v24 + mix(p_new, pb) - mix(p_v13, pb),   mix(p, pb) = clip(p * off_sched + (1 - p) * pb, 0)
    pb     = v24's real-taxi branch: 0.5 * W2 mix finals + 0.5 * the 27-fit seq pool finals
    p_v13  = the shipped Rome TabICL p (cache/strategy/tfm_P_lirf_ranking.npz)
    p_new  = TabICL (seeds 0-2, CPU, batch 1) on the 38 exp62 columns + car_apt_rate, car_apt_n, car_rate, car_n, fitted on
             all 627 Rome P rows of 2025; the carrier rates for 2026 come from ALL 2025 route-D rows (no 2026 label).
v24 has no Rome P row at the 0 clip (asserted), so the delta on v24 is exact up to the final rounding.

    HF_HUB_OFFLINE=1 python experiments/exp82_submit_v25.py --check identity    # p_new := p_v13 -> delta exactly 0
    HF_HUB_OFFLINE=1 python experiments/exp82_submit_v25.py --check summary
    HF_HUB_OFFLINE=1 python experiments/exp82_submit_v25.py --version 25
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(HERE, 'tail'))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load
from prc.submit import build, TEAM
from t06_carrier_stamp import carrier_features, KEYS, CCOLS
from v24_oof import build as build24, POOL


def fail(msg):
    sys.exit('FATAL: ' + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', choices=['identity', 'summary'], default=None)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.check is None and a.version is None:
        fail('pass --version N or --check identity|summary')

    # ---- 2025: Rome P training rows + carrier rates (month-OOF for training, as in t06) ----
    B = build24(); y, month, D, so25 = B['y'], B['month'], B['D'], B['so']
    Pidx, rome = B['Pidx'], B['rome']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'SCHED_TIME_UTC_mvt'])
    C, car = carrier_features(dd, y, so25, month, D)
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    num = [x for x in cand.columns if x not in KEYS and cand[x].dtype != object]
    Xb_tr = cand[num].astype(float).to_numpy()[rome]
    Xn_tr = np.hstack([Xb_tr, C.iloc[Pidx][CCOLS].to_numpy(float)[rome]])
    t_tr = (y[Pidx][rome] > 7200).astype(int)

    # ---- 2026: Rome P ranking rows, carrier rates from ALL of 2025 ----
    rk = load('ranking', phase='DEP', columns=[ID, 'APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True); ids, in_P = c['ids'], c['in_P'].astype(bool)
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('ranking order differs from the components')
    pos = np.flatnonzero(in_P)
    crk = pd.read_parquet(CACHE / 'exp62_cand_ranking.parquet').reset_index(drop=True)
    if len(crk) != len(pos) or not np.array_equal(crk.apt.astype(str).to_numpy(), rk.APT.astype(str).to_numpy()[pos]):
        fail('exp62_cand_ranking is not in the ranking P-row order')
    rr = crk.apt.astype(str).to_numpy() == 'LIRF'
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    rows = t1['pos']; p_v13 = t1['p_tabicl'].astype(float)
    if not np.array_equal(rows, pos[rr]):
        fail('the shipped Rome p is not in the same row order as the Rome P ranking rows')
    apt25 = dd.APT.astype(str).to_numpy()
    pop = D & (so25 > 1800) & np.isfinite(y); st = np.abs(so25 - y) <= 120
    t = pd.DataFrame({'car': car[pop], 'apt': apt25[pop], 's': st[pop]})
    ga, gc = t.groupby(['car', 'apt']).s.agg(['sum', 'size']), t.groupby('car').s.agg(['sum', 'size'])
    pa, p_all = t.groupby('apt').s.mean(), t.s.mean()
    car_rk = rk.FLIGHT_mvt.astype(str).str.upper().str.extract(r'^([A-Z]{3})')[0].fillna('???').to_numpy()[rows]
    q = pd.DataFrame({'car': car_rk, 'apt': 'LIRF'})
    ja, jc = q.join(ga, on=['car', 'apt']), q.join(gc, on='car')
    prior = q.apt.map(pa).fillna(p_all).to_numpy()
    Crk = np.column_stack([(ja['sum'].fillna(0) + 10 * prior) / (ja['size'].fillna(0) + 10), ja['size'].fillna(0),
                           (jc['sum'].fillna(0) + 10 * p_all) / (jc['size'].fillna(0) + 10), jc['size'].fillna(0)])
    Xb_rk = crk[num].astype(float).to_numpy()[rr]; Xn_rk = np.hstack([Xb_rk, Crk])

    import torch; torch.set_num_threads(a.threads)
    from tabicl import TabICLClassifier
    fitp = lambda Xtr, Xte: np.mean([TabICLClassifier(device='cpu', random_state=s, n_jobs=a.threads, batch_size=1)
                                     .fit(Xtr, t_tr).predict_proba(Xte)[:, 1] for s in range(3)], axis=0)
    p_base = fitp(Xb_tr, Xb_rk)
    if np.abs(p_base - p_v13).max() > 1e-3:
        fail(f'the refitted base TabICL does not reproduce the shipped v13 p (max diff {np.abs(p_base - p_v13).max():.4f})')
    p_new = p_v13 if a.check == 'identity' else fitp(Xn_tr, Xn_rk)
    np.savez(CACHE / 'tail' / 'v25_p_rome.npz', rows=rows, p_v13=p_v13, p_new=p_new, car=car_rk, Crk=Crk)

    # ---- serve: delta on v24 at the Rome P rows only ----
    so_rk = (rk.MVT_TIME_UTC_mvt - rk.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    mem = lambda folder, f: np.mean([np.load(CACHE / folder / f'{f}__seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)
    for folder in ['w2_final'] + [f'seq_{v}_final' for v in POOL]:
        if not np.array_equal(np.load(CACHE / folder / 'ranking_ids.npy', allow_pickle=True), ids):
            fail(f'{folder} is in a different ranking order')
    pb = 0.5 * mem('w2_final', 'mix') + 0.5 * np.mean([mem(f'seq_{v}_final', 'seq') for v in POOL], axis=0)
    mix = lambda p: np.clip(p * so_rk[rows] + (1 - p) * pb[rows], 0, None)
    v24 = pd.read_parquet(OUTPUTS / f'{TEAM}_v24.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
    if np.isnan(v24).any():
        fail('v24 misses ranking ids')
    if (v24[rows] <= 0).any():
        fail('a Rome P row sits at the 0 clip in v24 -- the delta is not valid there')
    delta = np.zeros(len(ids)); delta[rows] = mix(p_new) - mix(p_v13)
    new = np.rint(np.clip(v24 + delta, 0, 200000))
    if a.check == 'identity':
        print(f'IDENTITY check (p_new := p_v13): max |delta| {np.abs(delta).max():.2e}, rows changed {int((new != v24).sum())}')
        return
    d = new - v24; ch = d != 0
    print(f'refit base == shipped p (max diff {np.abs(p_base - p_v13).max():.1e}); p mean {p_v13.mean():.3f} -> {p_new.mean():.3f}; '
          f'|p_new - p_v13| median {np.median(np.abs(p_new - p_v13)):.3f}, max {np.abs(p_new - p_v13).max():.3f}')
    print(f'rows changed {int(ch.sum())} (all Rome P: {bool(np.all(np.isin(np.flatnonzero(ch), rows)))}); shift on them: mean {d[rows].mean():+.0f} s, '
          f'|d| median {np.median(np.abs(d[rows])):.0f}, p90 {np.percentile(np.abs(d[rows]), 90):.0f}, max {d.max():+.0f} / {d.min():+.0f}')
    tb = pd.DataFrame({'car': car_rk, 'd': d[rows], 'p_v13': p_v13, 'p_new': p_new})
    print(tb.groupby('car').agg(rows=('d', 'size'), mean_shift=('d', 'mean'), p_v13=('p_v13', 'mean'), p_new=('p_new', 'mean'))
          .sort_values('rows', ascending=False).head(10).round(2).to_string())
    if a.check == 'summary':
        return
    out = OUTPUTS / f'{TEAM}_v{a.version}.parquet'
    if out.exists() and not a.overwrite:
        fail(f'{out.name} exists; pass --overwrite to replace it')
    path = build(ids, new, version=a.version, clip=(0.0, 200000.0))
    got = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
    if not np.array_equal(got, new):
        fail('the written file differs from the in-memory build')
    print(f'v{a.version} written and read back identical; rows changed vs v24: {int((got != v24).sum())}')


if __name__ == '__main__':
    main()
