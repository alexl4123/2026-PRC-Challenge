"""Bet D: is the context effect just a simple neighbour aggregate? (label-free aggregates vs ctx - noctx)

On the subset's held-out bulk rows, compare how well v17's residual (y - v17) is explained by
  (a) ctx - noctx (seed means) -- what the set model's context adds, and
  (b) simple label-free aggregates over the SAME contexts (seq_net.build_context, K=64, +-90 min):
      prox_med_all   median NM proxy (MVT - AOBT_3) of departure neighbours
      prox_med_prev  the same over departures that took off in the 30 min before us (m11 'hpT'-like)
      prox_med_next  the same over departures that took off in the 30 min after us (look-ahead)
      prox_gap       our own proxy minus prox_med_all
      tin_med        median taxi-in of arrival neighbours;  n_dep_rwy_prev  deps on our runway in the 30 min before us
      ahead_pushed   departures that pushed back (AOBT_3) before us and take off after our AOBT_3 (queue ahead)
Reported: correlation with y - v17, and a 2-fold airport-day cross-fitted linear fit of y - v17 on the aggregates
(RMSE delta on top of v17), next to the same for ctx - noctx. If the aggregates match ctx - noctx, a hand-built column
in the members would capture the effect far more cheaply than a new member.

    python experiments/seq/s04_aggregates.py --tag LTFM-LFPG-EHAM__tr050608__te07__K64
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, TAIL_CAP
from prc.io import load
from prc.models import seq_net as S


def crossfit_lin(y, X, block, n_split=20):
    """RMSE delta of a least-squares fit of y on [1, X], 2-fold cross-fitted over blocks."""
    X = np.column_stack([np.ones(len(y)), np.nan_to_num(X)])
    ub = np.unique(block); rng = np.random.default_rng(0); out = []
    base = np.sqrt(np.mean(y ** 2))
    for _ in range(n_split):
        half = np.isin(block, rng.permutation(ub)[: len(ub) // 2])
        p = np.empty(len(y))
        for f, a in ((half, ~half), (~half, half)):
            w = np.linalg.lstsq(X[f], y[f], rcond=None)[0]
            p[a] = X[a] @ w
        out.append(np.sqrt(np.mean((y - p) ** 2)) - base)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='LTFM-LFPG-EHAM__tr050608__te07__K64')
    a = ap.parse_args()
    d = CACHE / 'seq' / 'subset' / a.tag
    R = np.load(d / 'rows.npz', allow_pickle=True)
    va, y, apt, v17, ids = R['va'], R['y'], R['apt'], R['v17'], R['ids']
    test_month = int(np.unique(R['month'][va])[0])
    airports = sorted(set(apt.astype(str)))
    full = load('training', columns=S.MV_COLS)
    keep = full.APT.astype(str).isin(airports).to_numpy() & (full.MVT_TIME_UTC_mvt.dt.month.to_numpy() == test_month)
    full = full[keep].reset_index(drop=True)
    mv, _ = S.movement_table(full, None)
    del full
    tgt = np.flatnonzero(mv['is_dep'])
    idx_va = np.flatnonzero(va)
    if not np.array_equal(mv['id'][tgt], ids[idx_va]):
        sys.exit('FATAL: held-out rows not aligned with the movement table')
    nb = S.build_context(mv, K=64, window=5400)
    j = np.clip(nb, 0, None); ok = nb >= 0
    T = mv['T'].astype(float)
    t0 = T[tgt, 0][:, None]; a3 = T[tgt, 1][:, None]
    tn, a3n = T[j, 0], T[j, 1]
    dep = mv['is_dep'][j] & ok; arr = ~mv['is_dep'][j] & ok
    prox = np.where(dep, tn - a3n, np.nan)
    tin = np.where(arr, T[j, 4] - tn, np.nan)
    rw_same = (mv['rwy'][j] == mv['rwy'][tgt][:, None]) & ok
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    agg = pd.DataFrame({
        'prox_med_all': np.nanmedian(prox, 1),
        'prox_med_prev': np.nanmedian(np.where((tn < t0) & (tn >= t0 - 1800), prox, np.nan), 1),
        'prox_med_next': np.nanmedian(np.where((tn > t0) & (tn <= t0 + 1800), prox, np.nan), 1),
        'tin_med': np.nanmedian(tin, 1),
        'n_dep_rwy_prev': (dep & rw_same & (tn < t0) & (tn >= t0 - 1800)).sum(1).astype(float),
        'ahead_pushed': (dep & (a3n < a3) & (tn > a3)).sum(1).astype(float),
    })
    agg['prox_gap'] = (t0[:, 0] - a3[:, 0]) - agg.prox_med_all
    yv, v = y[va], v17[va]
    k = yv <= TAIL_CAP
    c = np.mean([np.load(d / f'ctx__s{s}.npy') for s in (0, 1) if (d / f'ctx__s{s}.npy').exists()], 0)
    n = np.mean([np.load(d / f'noctx__s{s}.npy') for s in (0, 1) if (d / f'noctx__s{s}.npy').exists()], 0)
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_TIME_UTC_mvt'])
    day = dd.MVT_TIME_UTC_mvt.dt.strftime('%m%d').to_numpy()[R['frame_row'][va]]
    block = np.char.add(apt[va].astype(str), day.astype(str))
    r = (yv - v)[k]; bk = block[k]; ak = apt[va][k]
    delta = (c - n)[k]
    print(f'{a.tag}: held-out bulk rows {k.sum():,}')
    print('\ncorrelation with y - v17 (all | per airport):')
    cols = ['ctx - noctx'] + list(agg.columns)
    for nm in cols:
        x = delta if nm == 'ctx - noctx' else agg[nm].to_numpy()[k]
        def cc(m):
            f = np.isfinite(x[m]); return np.corrcoef(x[m][f], r[m][f])[0, 1] if f.sum() > 100 else np.nan
        print(f'  {nm:16s} {cc(np.ones(len(r), bool)):+.3f} | ' + '  '.join(f'{x_} {cc(ak == x_):+.3f}' for x_ in airports) +
              f'   (corr with ctx-noctx {np.corrcoef(np.nan_to_num(x), delta)[0, 1]:+.3f}, NaN {np.mean(~np.isfinite(x)):.0%})')
    print('\ncross-fitted linear fit of y - v17 (RMSE delta on top of v17), all | per airport:')
    A = agg.to_numpy()[k]
    A = np.column_stack([A, ~np.isfinite(A)])
    for nm, X in (('ctx - noctx', delta[:, None]), ('aggregates (7 + NaN flags)', A),
                  ('aggregates + ctx - noctx', np.column_stack([A, delta]))):
        print(f'  {nm:28s} {crossfit_lin(r, X, bk):+.2f} | ' + '  '.join(
            f'{x_} {crossfit_lin(r[ak == x_], X[ak == x_], bk[ak == x_]):+.2f}' for x_ in airports))


if __name__ == '__main__':
    main()
