"""seq hyper-parameter SCREEN evaluation (cluster/seq_sweep_array.sbatch): which configuration earns a full campaign.

Each configuration has 2 seeds x the 4 screening months (3, 6, 9, 12) in cache/cv/seqsw_<name>/. The baseline is the
shipped seq (cache/cv/seq, K 64, window 5400 s, d 64, 2 layers, 10 epochs), 3 seeds on the same months. Measured seed
noise of the baseline member on these months, outside P: mixture RMSE 185.68 / 185.16 / 184.30 (sd ~0.7 s).

PRE-REGISTERED RULE (written 2026-09-26 before any sweep fold exists):
  metric  = the member's own mixture RMSE on the screening months outside P, seeds averaged per row
            (bulk RMSE on bulk rows is reported alongside, not decisive);
  winner  = the configuration with the lowest metric;
  a full 12-month x 3-seed campaign is earned only if winner <= baseline (3-seed mean prediction) - 1.0 s, AND both of
            the winner's seeds individually beat the baseline's mean-prediction RMSE.
  With 6 configurations the best-of-6 noise alone is ~0.5-0.7 s on a 2-seed mean, hence the 1.0 s bar. The full
  campaign is then judged by the usual blend rule on the 8 HELD-OUT months (the screen never sees them).

    python experiments/seq/seq_sweep_eval.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import json
import numpy as np
from prc.config import CACHE, TAIL_CAP

SC = [3, 6, 9, 12]
NAMES = ['k128', 'w7200', 'k128w7200', 'd128', 'l3', 'ep15']


def load(folder, seeds, month, suffix='.npy'):
    o = np.full(len(month), np.nan)
    for m in SC:
        k = month == m
        o[k] = np.mean([np.load(folder / f'seq__seed{s:02d}__m{m:02d}{suffix}') for s in seeds], axis=0)
    return o


def main():
    b = np.load(CACHE / 'v11' / 'base.npz'); y, month, P = b['y'], b['month'], b['P']
    g = np.isin(month, SC) & ~P; gb = g & (y <= TAIL_CAP)
    r = lambda p, k: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))
    base = CACHE / 'cv' / 'seq'
    b_mix, b_bulk = load(base, [0, 1, 2], month), load(base, [0, 1, 2], month, '.bulk.npy')
    ref = r(b_mix, g)
    print(f'baseline (3 seeds): mixture {ref:.3f} | bulk {r(b_bulk, gb):.3f}')
    rows = {}
    for n in NAMES:
        f = CACHE / 'cv' / f'seqsw_{n}'
        seeds = [s for s in (0, 1) if all((f / f'seq__seed{s:02d}__m{m:02d}.npy').exists() for m in SC)]
        if not seeds:
            print(f'{n:10s} no complete seed yet')
            continue
        cfg = json.load(open(f / '_config.json')) if (f / '_config.json').exists() else {}
        mix, bulk = load(f, seeds, month), load(f, seeds, month, '.bulk.npy')
        per = [r(load(f, [s], month), g) for s in seeds]
        rows[n] = dict(metric=r(mix, g), bulk=r(bulk, gb), per_seed=per, seeds=seeds)
        print(f'{n:10s} seeds {seeds}: mixture {rows[n]["metric"]:.3f} ({rows[n]["metric"] - ref:+.3f}) | bulk '
              f'{rows[n]["bulk"]:.3f} | per seed ' + ' '.join(f'{v:.3f}' for v in per) +
              f' | K {cfg.get("K")} window {cfg.get("window")} d {cfg.get("d")} layers {cfg.get("layers")} epochs {cfg.get("epochs")}')
    full = {n: v for n, v in rows.items() if len(v['seeds']) == 2}
    if len(full) < len(NAMES):
        print(f'\nINCOMPLETE: {len(full)} of {len(NAMES)} configurations have both seeds -- no decision yet')
        return
    win = min(full, key=lambda n: full[n]['metric'])
    ok = full[win]['metric'] <= ref - 1.0 and all(v < ref for v in full[win]['per_seed'])
    print(f'\nwinner {win}: {full[win]["metric"] - ref:+.3f} s vs baseline -> '
          + ('EARNS a full campaign' if ok else 'does NOT clear the 1.0 s bar -- keep the shipped configuration'))


if __name__ == '__main__':
    main()
