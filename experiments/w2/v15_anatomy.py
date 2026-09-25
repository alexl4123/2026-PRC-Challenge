"""Where does the error of the CURRENT structure (v15) live? 2025 leave-one-month-out.

The last error anatomy (strategy s13) was for v11; v14/v15 replaced most of the structure. This
rebuilds the v15 structure OOF exactly as experiments/w2/w3_eval.py does (blend6) and splits its
squared error (12 months and the scored months) by route cell, airport and label band, so the next
experiment aims at where the seconds are. For each cell it reports the share of total SSE and the
all-flights RMSE drop if that cell's error were halved ("half-fix"), a scale-free way to compare
cells of very different sizes.

    python experiments/w2/v15_anatomy.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from prc.config import CACHE
from w2_eval import oof, NEW, FAMS
from w2_routeD_P import bulk_oof


def main():
    b = np.load(CACHE / 'v11' / 'base.npz'); P, D, y, month = b['P'], b['D'], b['y'], b['month']
    W = json.load(open(CACHE / 'w2' / 'w2_weights.json'))
    mem = {f: oof(NEW, f, [0, 1, 2], month) for f in FAMS}
    v11 = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    v9 = np.load(CACHE / 'major' / 'v9s_oof.npy').astype(float)
    v10 = np.clip(np.load(CACHE / 'major' / 'm11d_oof_trailD.npy').astype(float), 0, None)
    w4 = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w4['Pidx']; rome = w4['apt'] == 'LIRF'
    p14 = np.where(rome, w4['p_own'], w4['p_pool']); p_cur = w4['p_cur']
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['APT', 'off_sched', 'off_aobt3'])
    so_all = dd.off_sched.to_numpy(); apt = dd.APT.astype(str).to_numpy()
    so = so_all[Pidx]
    pb_old = np.load(CACHE / 'exp62_pb.npy').astype(float)[Pidx]
    pb_new = 0.5 * bulk_oof('mix', [0, 1, 2], month)[Pidx] + 0.5 * bulk_oof('nn', [0, 1, 2], month)[Pidx]
    mixP = lambda p, pb: np.clip(p * so + (1 - p) * pb, 0, None)
    A, DnP = ~D & ~P, D & ~P
    f = v11.copy()
    for g, key in ((A, 'routes A/B/C'), (DnP, 'route D')):
        blend = sum(W['weights'][key][m] * mem[m] for m in FAMS)
        f[g] = np.clip(blend[g] + ((v10 - v9)[g] if key == 'route D' else 0.0), 0, None)
    f[Pidx] = v11[Pidx] + (mixP(p14, pb_new) - mixP(p_cur, pb_old))
    np.save(CACHE / 'w2' / 'v15s_oof.npy', f.astype(np.float32))

    e2 = (y - f) ** 2
    S = np.isin(month, [1, 7])
    band = np.select([y <= 1800, y <= 3600, y <= 7200], ['y<=30min', '30-60min', '1-2h'], '>2h')
    cell = np.where(P, 'P', np.where(D, 'D (not P)', 'A/B/C'))

    def table(title, keys, scope):
        tot = e2[scope].sum(); n = scope.sum(); rm = np.sqrt(tot / n)
        print(f'\n{title} -- RMSE {rm:.2f} over {n:,} flights')
        print(f"  {'cell':26s}{'rows':>10s}{'row %':>8s}{'SSE %':>8s}{'cell RMSE':>11s}{'half-fix':>10s}")
        rows = []
        for k in sorted(set(keys[scope])):
            m = scope & (keys == k)
            s = e2[m].sum()
            half = rm - np.sqrt((tot - 0.75 * s) / n)        # halving the cell's RMSE = removing 75% of its SSE
            rows.append((s, k, m.sum(), np.sqrt(s / m.sum()), half))
        for s, k, nk, crm, half in sorted(rows, reverse=True):
            print(f'  {k:26s}{nk:10,}{100 * nk / n:8.2f}{100 * s / tot:8.1f}{crm:11.1f}{half:+10.2f}')

    for scope_name, scope in (('12 MONTHS', np.ones(len(y), bool)), ('SCORED MONTHS (1, 7)', S)):
        table(f'{scope_name}: route cell x label band', np.char.add(np.char.add(cell.astype(str), ' | '), band), scope)
        table(f'{scope_name}: airport x route cell', np.char.add(np.char.add(apt, ' | '), cell.astype(str)), scope)


if __name__ == '__main__':
    main()
