"""Where v21's error lives (2025 OOF in the v21 structure) and where it is EXPECTED to live in 2026.

v21 structure: 7 adsb2 members + seq (15 epochs, seeds 0-2) outside P (weights refitted = cache/seq/swap_seqsw_ep15.json),
route-D corrector x0.5, zone P = TabICL p with pb = 0.5 W2 mix + 0.5 seq15 bulk. Saved as cache/w2/v21s_oof.npy.
Then, with the drivers-review classes (experiments/drivers_0926/d02_decompose.py):
  (1) SSE share by mechanism (day-scale artefacts, tail classes, P false alarms, stamping, ordinary bulk), 12 months
      and Jan+Jul, with the oracle "zerofix" (RMSE drop if that class were predicted perfectly);
  (2) expected 2026 SSE by airport x route x ADS-B status (2025 12-month cell MSE x 2026 cell counts, day-scale rows
      excluded) -- exactly d03_extra.py (a), with v21 in place of v17.

    python experiments/tail/t00_v21_anatomy.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'drivers_0926'))
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from d02_decompose import mechanism, adsb_status

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3', 'seq']
SCREEN = [3, 6, 9, 12]
OUT = CACHE / 'tail'
pd.set_option('display.width', 220)


def oof(folder, fam, seeds, month, suffix='.npy'):
    o = np.zeros(len(month))
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


def v21_oof():
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    M = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2], month) for f in FAMS[:-1]}
    M['seq'] = oof(CACHE / 'cv' / 'seqsw_ep15', 'seq', [0, 1, 2], month)
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    so = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['off_sched']).off_sched.to_numpy()[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    pbP = 0.5 * oof(CACHE / 'cv' / 'w2', 'mix', [0, 1, 2], month, '.bulk.npy')[Pidx] + \
        0.5 * oof(CACHE / 'cv' / 'seqsw_ep15', 'seq', [0, 1, 2], month, '.bulk.npy')[Pidx]
    fit = np.isin(month, SCREEN) & ~P
    W = {}
    for key, g in (('routes A/B/C', ~D), ('route D', D)):
        w, _ = nnls(np.column_stack([M[f] for f in FAMS])[fit & g], y[fit & g])
        W[key] = dict(zip(FAMS, np.round(w, 4).tolist()))
    ref = json.load(open(CACHE / 'seq' / 'swap_seqsw_ep15.json'))['weights_new']
    dw = max(abs(W[k][f] - ref[k].get(f, 0.0)) for k in W for f in FAMS)
    if dw > 1e-3:
        sys.exit(f'FATAL: refitted weights differ from the shipped v21 weights by {dw}')
    f = v11.copy()
    for g, key in ((~D & ~P, 'routes A/B/C'), (D & ~P, 'route D')):
        bl = sum(W[key][m] * M[m] for m in FAMS)
        f[g] = np.clip(bl[g] + (0.5 * (v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
    f[Pidx] = v11[Pidx] + (mixP(p14, pbP) - mixP(p_cur, pb_old))
    return f, y, month


def main():
    OUT.mkdir(exist_ok=True)
    f, y, month = v21_oof()
    np.save(CACHE / 'w2' / 'v21s_oof.npy', f.astype(np.float32))
    r = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet')
    k = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet')
    if not np.allclose(r.y.to_numpy(float), y, equal_nan=True):
        sys.exit('FATAL: rows_training not aligned')
    e2 = (y - f) ** 2; e17 = (y - r.pred.to_numpy()) ** 2
    r['mech'] = mechanism(r); r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
    day = r.mech.to_numpy() == 'day'
    for scope_name, sc in (('12 months', np.ones(len(y), bool)), ('Jan+Jul 2025', np.isin(month, [1, 7]))):
        tot, n = e2[sc].sum(), sc.sum(); rm = np.sqrt(tot / n)
        print(f'\n== v21 OOF, {scope_name}: RMSE {rm:.2f} (v17 {np.sqrt(e17[sc].mean()):.2f}); without day-scale rows '
              f'{np.sqrt(e2[sc & ~day].mean()):.2f}')
        g = pd.DataFrame({'m': r.mech.to_numpy()[sc], 'e': e2[sc], 'e17': e17[sc]}).groupby('m').agg(
            rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot), rmse=('e', lambda v: np.sqrt(v.mean())),
            rmse_v17=('e17', lambda v: np.sqrt(v.mean())), sse=('e', 'sum'))
        g['zerofix'] = rm - np.sqrt((tot - g.sse) / n)
        print(g.drop(columns='sse').sort_values('sse_pct', ascending=False).round(2).to_string())

    def cell(x):
        sob = pd.cut(x.off_sched, [-np.inf, 0, 900, 1800, 3600, 7200, np.inf], labels=False).fillna(-1).astype(int).astype(str)
        return x.apt.astype(str) + '|' + x.route.astype(str) + '|' + x.adsb.astype(str), sob
    c25, b25 = cell(r); c26, b26 = cell(k)
    fine25, fine26 = (c25 + '|' + b25).to_numpy(), (c26 + '|' + b26).to_numpy()
    res = {}
    for nm, ee in (('v21', e2), ('v17', e17)):
        mse = pd.Series(ee[~day]).groupby(fine25[~day]).mean(); msec = pd.Series(ee[~day]).groupby(c25.to_numpy()[~day]).mean()
        m26 = pd.Series(fine26).map(mse); miss = m26.isna().to_numpy()
        m26[miss] = pd.Series(c26.to_numpy()[miss]).map(msec).to_numpy()
        res[nm] = m26.fillna(m26.mean()).to_numpy()
    exp = pd.DataFrame({'apt': k.apt.astype(str).to_numpy(), 'route': k.route.astype(str).to_numpy(),
                        'adsb': k.adsb.to_numpy(), 'e': res['v21'], 'e17': res['v17']})
    tot = exp.e.sum()
    print(f'\n== EXPECTED 2026 (day-scale excluded): RMSE v21 {np.sqrt(exp.e.mean()):.1f}  (v17 {np.sqrt(exp.e17.mean()):.1f})')
    ta = exp.groupby(['apt', 'route', 'adsb']).agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot),
                                                    rmse=('e', lambda v: np.sqrt(v.mean())),
                                                    rmse_v17=('e17', lambda v: np.sqrt(v.mean()))).sort_values('sse_pct', ascending=False)
    ta['cum_pct'] = ta.sse_pct.cumsum()
    print(ta.head(15).round(1).to_string())
    tb = exp.groupby('apt').agg(rows=('e', 'size'), sse_pct=('e', lambda v: 100 * v.sum() / tot),
                                rmse=('e', lambda v: np.sqrt(v.mean())), rmse_v17=('e17', lambda v: np.sqrt(v.mean())))
    print('\nby airport:'); print(tb.sort_values('sse_pct', ascending=False).round(1).to_string())
    ta.to_csv(OUT / 't00_expected_2026_cells_v21.csv'); tb.to_csv(OUT / 't00_expected_2026_airports_v21.csv')


if __name__ == '__main__':
    main()
