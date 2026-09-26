"""routed_0926 step 3 (POST-HOC, written after r02 showed E1 and E2 disagreeing on the EHAM-January zone-P rows; NOT
part of the pre-registered rule): which implied label deserves trust, and what the candidates are worth live [E].

1. LIVE CALIBRATION. v9 -> v10 changed ONLY the corrector rows, and its live effect is measured: 297.3414 -> 292.3393
   (-5.00 s, all 344,841 ranking rows scored), i.e. dSSE_live = N (292.3393^2 - 297.3414^2). Each implied label L
   predicts dSSE = sum over covered rows of (L - v10)^2 - (L - v9)^2, scaled to all corrector rows per
   (airport, month, P or not) cell by rows / covered rows. The estimator whose prediction lands near the measured value
   is the one to trust for this event. (Our own leaderboard scores only; no external scored-month information.)
2. The same estimators' bias on the 2025 labelled rows with the remote-rest pattern (the de-icing signature).
3. EHAM January 2026 by local day: rows, covered rows, base, corrector, E1, E2, route-A calibration rows.
4. Expected live effect [E] of b / c / d vs v17 under E1 and E2, same scaling, as dRMSE at the v17 live 251.12.

    python experiments/routed_0926/r03_calibrate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import peak_rss_gb
from r02_2026 import adsb, codes

OUT = CACHE / 'routed_0926'
N = 344841
LIVE = {'v9': 297.3414, 'v10': 292.3393, 'v17': 251.12}
pd.set_option('display.width', 250)


def scaled_dsse(L, new, old, cov, R, cellkey):
    """Pseudo dSSE on covered rows, scaled per cell to all rows of R in that cell. Returns (covered-only, scaled)."""
    k = R & cov & np.isfinite(L)
    d = np.where(k, (L - new) ** 2 - (L - old) ** 2, 0.0)
    tot_cov, tot_sc = d.sum(), 0.0
    for c in np.unique(cellkey[R]):
        m = R & (cellkey == c); mk = m & k
        if mk.sum() >= 5:
            tot_sc += d[mk].sum() * m.sum() / mk.sum()
        # cells with < 5 covered rows contribute nothing (reported as uncovered share)
    return tot_cov, tot_sc


def drmse(dsse, base):
    return np.sqrt(base ** 2 + dsse / N) - base


def main():
    z6 = np.load(OUT / 'arrays_2026.npz', allow_pickle=True); z5 = np.load(OUT / 'arrays_2025.npz', allow_pickle=True)
    imp = np.load(OUT / 'r02_implied.npz'); E1_6, E2_6, E1_5, E2_5 = imp['E1_6'], imp['E2_6'], imp['E1_5'], imp['E2_5']
    ids = z6['ids']; R6, P6, m6, apt6 = z6['R'], z6['P'], z6['month'], z6['apt']
    c6 = codes(apt6, m6, adsb('ranking', ids))
    cell = np.char.add(np.char.add(apt6.astype(str), m6.astype(str)), np.where(P6, 'P', 'D'))
    ver = lambda v: pd.read_parquet(OUTPUTS / f'understanding-accordion_v{v}.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy(float)
    v9, v10 = ver(9), ver(10)
    moved = np.abs(v10 - v9) > 0
    print(f'v9 -> v10: rows changed {moved.sum():,}, all inside R: {bool((moved <= R6).all())}; covered share of changed rows '
          f'{c6["cov"][moved].mean():.3f}')
    live = N * (LIVE['v10'] ** 2 - LIVE['v9'] ** 2)
    print(f'\n1. LIVE CALIBRATION  measured dSSE(v10 - v9) = {live:+.3e} s^2  (dRMSE {LIVE["v10"] - LIVE["v9"]:+.2f} s)')
    for nm, L in (('E1', E1_6), ('E2', E2_6)):
        for sc, Rm in (('all R', R6), ('R outside P', R6 & ~P6), ('R inside P', R6 & P6)):
            cv, s = scaled_dsse(L, v10, v9, c6['cov'], Rm, cell)
            print(f'  {nm} {sc:12s}: covered-only {cv:+.3e}, scaled {s:+.3e}  -> predicted dRMSE {drmse(s, LIVE["v9"]):+.2f} s')
    # (A second calibration point was tried and dropped: v13 -> v14 also swapped the W2 members on A/B/C -- 333,810 rows
    #  changed, not only the 1,525 P rows -- so its live -7.58 s is not attributable to route D / P.)
    # the whole-cell view: EHAM January
    for sc, Rm in (('EHAM-1 outside P', R6 & ~P6 & (apt6 == 'EHAM') & (m6 == 1)), ('EHAM-1 inside P', R6 & P6 & (apt6 == 'EHAM') & (m6 == 1))):
        k = Rm & c6['cov']
        print(f'  {sc}: rows {Rm.sum()}, covered {k.sum()}; mean v9 {v9[k].mean():.0f}, v10 {v10[k].mean():.0f}, E1 {E1_6[k].mean():.0f}, '
              f'E2 {np.nanmean(E2_6[k]):.0f}')

    print('\n2. Implied-label bias on 2025 labelled corrector rows (mean L - y, RMSE), remote-rest pattern vs the rest')
    c5 = codes(z5['apt'], z5['month'], adsb('training', pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=[ID])[ID].to_numpy()))
    y = z5['y']
    for sc, S in (('S1 D outside P', z5['R'] & ~z5['P']), ('S2 P & R', z5['R'] & z5['P'])):
        for nm, k in (('remote-rest', S & c5['rr']), ('remote-rest winter', S & c5['rr'] & np.isin(z5['month'], [12, 1, 2])),
                      ('other covered', S & c5['cov'] & ~c5['rr'])):
            k1 = k & np.isfinite(E1_5); k2 = k & np.isfinite(E2_5)
            print(f'  {sc:15s} {nm:19s} n={k1.sum():5d}: E1 {np.mean(E1_5[k1] - y[k1]):+6.0f} / {np.sqrt(np.mean((E1_5[k1] - y[k1]) ** 2)):5.0f}   '
                  f'E2 (n={k2.sum()}) {np.mean(E2_5[k2] - y[k2]):+6.0f} / {np.sqrt(np.mean((E2_5[k2] - y[k2]) ** 2)):5.0f}   '
                  f'mean y {y[k1].mean():.0f}')

    print('\n3. EHAM January 2026 by local date (corrector rows)')
    date6 = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet', columns=['date_loc']).date_loc.to_numpy()
    oa = pd.read_parquet(CACHE / 'major' / 'dep_ranking.parquet', columns=['off_aobt3']).off_aobt3.to_numpy(float)
    A6 = c6['cov'] & np.isfinite(oa) & (oa > 0) & (oa <= 7200)
    e = (apt6 == 'EHAM') & (m6 == 1)
    rows = []
    for dte in np.unique(date6[e & R6]):
        kd = e & (date6 == dte)
        for sc, Sm in (('D', R6 & ~P6), ('P', R6 & P6)):
            k = kd & Sm
            if k.sum() == 0:
                continue
            kc = k & c6['cov']
            rows.append(dict(date=dte, scope=sc, rows=k.sum(), cov=kc.sum(), rr=int((k & c6['rr']).sum()),
                             base=z6['base'][k].mean(), corr_old=z6['corr_old'][k].mean(),
                             E1=E1_6[kc].mean() if kc.any() else np.nan, E2=np.nanmean(E2_6[kc]) if kc.any() else np.nan,
                             base_cov=z6['base'][kc].mean() if kc.any() else np.nan,
                             routeA_cov=int((kd & A6).sum()), routeA_proxy_med=np.median(oa[kd & A6]) if (kd & A6).any() else np.nan,
                             routeA_adsb_med=np.median(c6['taxi'][kd & A6]) if (kd & A6).any() else np.nan))
    print(pd.DataFrame(rows).round(0).to_string(index=False))

    print('\n4. Expected live effect [E] of each candidate vs v17 (scaled pseudo dSSE; dRMSE at v17 live 251.12)')
    cn6 = np.load(OUT / 'refit_2026.npz', allow_pickle=True)['corr_new']
    base, co, v17 = z6['base'], z6['corr_old'], z6['v17']
    for sc, S in (('S1 D outside P', R6 & ~P6), ('S2 P & R', R6 & P6), ('S1+S2', R6)):
        for cnm, c in (('b drop', 0 * co), ('c half', 0.5 * co), ('d refit', cn6)):
            f = v17.copy(); f[S] = np.clip(base[S] + c[S], 0, None)
            out = []
            for nm, L in (('E1', E1_6), ('E2', E2_6)):
                cv, s = scaled_dsse(L, f, v17, c6['cov'], S, cell)
                out.append(f'{nm} {drmse(s, LIVE["v17"]):+.2f} s (covered only {drmse(cv, LIVE["v17"]):+.2f})')
            print(f'  {sc:15s} {cnm:8s}: ' + ' | '.join(out))
    print(f'\npeak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
