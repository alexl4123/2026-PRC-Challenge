"""Descriptive checks on t06(a) -- the carrier stamping rate in Rome's zone-P TabICL -- after it FAILED its rule
(12m -1.52 but Jan +0.06 and concentrated). NO decision is taken here; these checks describe the failure mode:
  1. anatomy: which Rome P rows gain / lose, by the carrier's 2025 sample size, tail vs bulk, direction of the p change,
     day-scale rows; the top losers and gainers;
  2. stability: is a carrier's route-D stamping rate stable within 2025 (H1 = Jan-Jun vs H2 = Jul-Dec)?
  3. transfer (label-free on the 2026 side): do the 181 Rome P ranking rows have carriers with solid 2025 estimates, and
     how far would p and the served value move (TabICL base vs base + carrier, both fitted on all 627 Rome P rows)?
    HF_HUB_OFFLINE=1 python experiments/tail/t07_carrier_checks.py --threads 3
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from prc.config import CACHE, ID
from prc.io import load
from v24_oof import build
from t06_carrier_stamp import carrier_features, KEYS, CCOLS

pd.set_option('display.width', 220)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=3); a = ap.parse_args()
    t0 = time.time()
    B = build(); y, month, D, f24, so = B['y'], B['month'], B['D'], B['f'], B['so']
    Pidx, rome, p14, pbP, P_rest = B['Pidx'], B['rome'], B['p14'], B['pbP'], B['P_rest']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'SCHED_TIME_UTC_mvt'])
    C, car = carrier_features(dd, y, so, month, D)
    ps = np.load(CACHE / 'tail' / 't06a_seeds.npy'); p_new = ps.mean(0); p_base = np.load(CACHE / 'tfm_f2' / 'base.npy')
    mixP = lambda p: np.clip(p * so[Pidx] + (1 - p) * pbP, 0, None)
    fb, fn = P_rest + mixP(p_base), P_rest + mixP(p_new)
    yP = y[Pidx]; R = rome
    d = (yP - fn) ** 2 - (yP - fb) ** 2                     # per-row change in squared error (negative = better)
    N, rm = len(y), np.sqrt(np.mean((y - f24) ** 2))
    to_rmse = lambda s: s / (2 * N * rm)                     # first-order RMSE effect of an SSE change
    nP = C['car_apt_n'].to_numpy()[Pidx]
    day = (yP > 43200) & (np.abs(yP - so[Pidx]) > 600)
    T = pd.DataFrame({'car': car[Pidx], 'n': nP, 'rate': C['car_apt_rate'].to_numpy()[Pidx], 'p_base': p_base, 'p_new': p_new,
                      'y': yP, 'so': so[Pidx], 'is_tail': yP > 7200, 'day': day, 'up': p_new > p_base, 'd': d, 'm': month[Pidx]})[R]
    print(f'== 1. ANATOMY, Rome P rows {len(T)} (tails {int(T.is_tail.sum())}, day-scale {int(T.day.sum())}): total dSSE {T.d.sum():.3e} '
          f'= {to_rmse(T.d.sum()):+.2f} s of 12-month RMSE; gains {int((T.d < 0).sum())} rows {to_rmse(T.d[T.d < 0].sum()):+.2f} s, '
          f'losses {int((T.d > 0).sum())} rows {to_rmse(T.d[T.d > 0].sum()):+.2f} s')
    T['nbin'] = pd.cut(T.n, [-1, 0, 9, 49, np.inf], labels=['n=0', '1-9', '10-49', '50+'])
    for key in ('nbin', 'is_tail', 'up', 'day'):
        g = T.groupby(key, observed=True).agg(rows=('d', 'size'), rmse_effect=('d', lambda v: to_rmse(v.sum())),
                                              gain_rows=('d', lambda v: int((v < 0).sum())), loss_rows=('d', lambda v: int((v > 0).sum())))
        print(f'  by {key}:'); print(g.round(2).to_string())
    cols = ['car', 'n', 'rate', 'p_base', 'p_new', 'y', 'so', 'm', 'day']
    for nm, o in (('top 8 LOSERS', T.sort_values('d', ascending=False)), ('top 8 GAINERS', T.sort_values('d'))):
        x = o.head(8).assign(rmse_s=lambda q: q.d.map(to_rmse))
        print(f'  {nm}:'); print(x[cols + ['rmse_s']].round(3).to_string(index=False))

    print('\n== 2. STABILITY of the route-D stamping rate within 2025 (off_sched > 1800, H1 = Jan-Jun vs H2 = Jul-Dec)')
    apt = dd.APT.astype(str).to_numpy()
    pop = D & (so > 1800) & np.isfinite(y); st = np.abs(so - y) <= 120
    for scope, k in (('all airports (carrier x airport)', pop), ('LIRF', pop & (apt == 'LIRF'))):
        t = pd.DataFrame({'key': pd.Series(car[k]) + '|' + pd.Series(apt[k]), 'h2': month[k] > 6, 's': st[k]})
        g = t.groupby(['key', 'h2']).s.agg(['mean', 'size']).unstack('h2')
        g = g[(g[('size', False)] >= 20) & (g[('size', True)] >= 20)]
        if len(g) >= 3:
            w = np.minimum(g[('size', False)], g[('size', True)]).to_numpy(float)
            a1, a2 = g[('mean', False)].to_numpy(), g[('mean', True)].to_numpy()
            cw = np.cov(a1, a2, aweights=w); r_w = cw[0, 1] / np.sqrt(cw[0, 0] * cw[1, 1])
            print(f'  {scope}: {len(g)} carriers with >= 20 rows in both halves; corr(H1, H2) {np.corrcoef(a1, a2)[0, 1]:+.3f}, '
                  f'weighted {r_w:+.3f}; rates H1 {a1.min():.2f}-{a1.max():.2f}')
            print('   ' + '  '.join(f'{k.split("|")[0]} {x1:.2f}/{x2:.2f}' for k, x1, x2 in sorted(zip(g.index, a1, a2), key=lambda z: -z[1])[:12]))
        else:
            print(f'  {scope}: only {len(g)} carriers with >= 20 rows in both halves -- too few to say')

    print('\n== 3. TRANSFER to the 181 Rome P ranking rows (label-free)')
    rk = load('ranking', phase='DEP', columns=[ID, 'APT', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt'])
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True); ids, in_P, et = c['ids'], c['in_P'].astype(bool), c['et']
    assert np.array_equal(rk[ID].to_numpy(), ids)
    crk = pd.read_parquet(CACHE / 'exp62_cand_ranking.parquet').reset_index(drop=True)
    pos = np.flatnonzero(in_P)
    assert len(crk) == len(pos) and np.array_equal(crk.apt.astype(str).to_numpy(), rk.APT.astype(str).to_numpy()[pos])
    rr = crk.apt.astype(str).to_numpy() == 'LIRF'
    car_rk = rk.FLIGHT_mvt.astype(str).str.upper().str.extract(r'^([A-Z]{3})')[0].fillna('???').to_numpy()[pos]
    so_rk = (rk.MVT_TIME_UTC_mvt - load('ranking', phase='DEP', columns=['SCHED_TIME_UTC_mvt']).SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()[pos]
    t = pd.DataFrame({'car': car[pop], 'apt': apt[pop], 's': st[pop]})
    ga = t.groupby(['car', 'apt']).s.agg(['sum', 'size']); gc = t.groupby('car').s.agg(['sum', 'size'])
    pa = t.groupby('apt').s.mean(); p_all = t.s.mean()
    q = pd.DataFrame({'car': car_rk, 'apt': crk.apt.astype(str).to_numpy()})
    ja, jc = q.join(ga, on=['car', 'apt']), q.join(gc, on='car')
    prior = q.apt.map(pa).fillna(p_all).to_numpy()
    Crk = pd.DataFrame({'car_apt_rate': (ja['sum'].fillna(0) + 10 * prior) / (ja['size'].fillna(0) + 10), 'car_apt_n': ja['size'].fillna(0),
                        'car_rate': (jc['sum'].fillna(0) + 10 * p_all) / (jc['size'].fillna(0) + 10), 'car_n': jc['size'].fillna(0)})
    n_rk = Crk.car_apt_n.to_numpy()[rr]
    print(f'  2026 Rome P rows: {rr.sum()}; carriers with 2025 LIRF route-D sample n = 0: {np.mean(n_rk == 0):.2f}, 1-9: '
          f'{np.mean((n_rk > 0) & (n_rk < 10)):.2f}, 10-49: {np.mean((n_rk >= 10) & (n_rk < 50)):.2f}, 50+: {np.mean(n_rk >= 50):.2f} '
          f'(2025 Rome P rows, OOF: n=0 {np.mean(T.n == 0):.2f}, 50+ {np.mean(T.n >= 50):.2f})')
    print('  top 2026 carriers: ' + ', '.join(f'{k} {v}' for k, v in pd.Series(car_rk[rr]).value_counts().head(8).items()))
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    num = [x for x in cand.columns if x not in KEYS and cand[x].dtype != object]
    Xb_tr, Xb_rk = cand[num].astype(float).to_numpy()[rome], crk[num].astype(float).to_numpy()[rr]
    Xn_tr = np.hstack([Xb_tr, C.iloc[Pidx][CCOLS].to_numpy(float)[rome]]); Xn_rk = np.hstack([Xb_rk, Crk[CCOLS].to_numpy(float)[rr]])
    t_tr = (y[Pidx][rome] > 7200).astype(int)
    import torch; torch.set_num_threads(a.threads)
    from tabicl import TabICLClassifier
    fitp = lambda Xtr, Xte: np.mean([TabICLClassifier(device='cpu', random_state=s, n_jobs=a.threads, batch_size=1)
                                     .fit(Xtr, t_tr).predict_proba(Xte)[:, 1] for s in range(3)], axis=0)
    pb_rk, pn_rk = fitp(Xb_tr, Xb_rk), fitp(Xn_tr, Xn_rk)
    ship = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)['p_tabicl'].astype(float)
    pbR = np.mean([np.load(CACHE / f'seq_{v}_final' / f'seq__seed{s:02d}.npy') for v in ('ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3') for s in range(3)], axis=0)
    w2mix = np.mean([np.load(CACHE / 'w2_final' / f'mix__seed{s:02d}.npy') for s in range(3)], axis=0)
    pbv = (0.5 * w2mix + 0.5 * pbR)[pos][rr]
    dv = (pn_rk - pb_rk) * (so_rk[rr] - pbv)                 # change of the served value, seconds
    dv25 = (T.p_new - T.p_base).to_numpy() * (T.so.to_numpy() - pbP[R])
    print(f'  p base (refit) vs shipped v13 p: mean |diff| {np.mean(np.abs(pb_rk - ship)):.3f}; p base mean {pb_rk.mean():.3f}, + carrier mean {pn_rk.mean():.3f}')
    for nm, v in (('2026 ranking', dv), ('2025 OOF', dv25)):
        print(f'  {nm}: served-value change |dv| median {np.median(np.abs(v)):.0f} s, p90 {np.percentile(np.abs(v), 90):.0f} s, '
              f'rows with |dv| > 1000 s: {np.mean(np.abs(v) > 1000):.2f}, mean dv {v.mean():+.0f} s')
    print(f'done {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
