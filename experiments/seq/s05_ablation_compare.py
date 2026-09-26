"""Bet D: context ablations vs the full context, seed-matched (seed 0), on the subset's held-out bulk rows.

For each ablation folder <base>__<ablation>/ctx__s0.npy: RMSE all / per airport next to the full-context ctx s0 and
noctx s0 of <base>, the share of the context effect (noctx - ctx) the ablation keeps, and ablation - full with an
airport-day block bootstrap CI.

    python experiments/seq/s05_ablation_compare.py --base LTFM-LFPG-EHAM__tr050608__te07__K64 --ablations nostand past
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, TAIL_CAP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s02_subset_eval import block_boot, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='LTFM-LFPG-EHAM__tr050608__te07__K64')
    ap.add_argument('--ablations', nargs='+', default=['nostand', 'past'])
    a = ap.parse_args()
    d = CACHE / 'seq' / 'subset' / a.base
    R = np.load(d / 'rows.npz', allow_pickle=True)
    va = R['va']; y = R['y'][va]; apt = R['apt'][va].astype(str)
    k = y <= TAIL_CAP
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_TIME_UTC_mvt'])
    day = dd.MVT_TIME_UTC_mvt.dt.strftime('%m%d').to_numpy()[R['frame_row'][va]]
    block = np.char.add(apt, day.astype(str))
    full, nc = np.load(d / 'ctx__s0.npy').astype(float), np.load(d / 'noctx__s0.npy').astype(float)
    arms = {'ctx (full context)': full, 'noctx': nc}
    for ab in a.ablations:
        f = CACHE / 'seq' / 'subset' / f'{a.base}__{ab}' / 'ctx__s0.npy'
        if f.exists():
            arms[f'ctx {ab}'] = np.load(f).astype(float)
        else:
            print(f'missing {f}')
    yk, ak, bk = y[k], apt[k], block[k]
    apts = sorted(set(ak))
    print(f'{a.base}, seed 0, held-out bulk rows {k.sum():,}')
    print(f"  {'':22s}{'all':>8s}" + ''.join(f'{x:>8s}' for x in apts) + '   kept share of the context effect (all | per airport)')
    eff = lambda p, m: rmse(yk[m], nc[k][m]) - rmse(yk[m], p[k][m])
    for nm, p in arms.items():
        line = f'  {nm:22s}{rmse(yk, p[k]):8.2f}' + ''.join(f'{rmse(yk[ak == x], p[k][ak == x]):8.2f}' for x in apts)
        if nm.startswith('ctx '):
            m_all = np.ones(len(yk), bool)
            line += f'   {eff(p, m_all) / eff(full, m_all):5.0%} | ' + '  '.join(
                f'{x} {eff(p, ak == x) / eff(full, ak == x):4.0%}' if abs(eff(full, ak == x)) > 0.5 else f'{x} n/a'
                for x in apts)
        print(line)
    for nm, p in arms.items():
        if nm.startswith('ctx ') and nm != 'ctx (full context)':
            print(f'  {nm} - full context: {"%+.2f [%+.2f, %+.2f]" % block_boot(yk, p[k], full[k], bk)}; ' + '  '.join(
                f'{x} {rmse(yk[ak == x], p[k][ak == x]) - rmse(yk[ak == x], full[k][ak == x]):+.2f}' for x in apts))


if __name__ == '__main__':
    main()
