"""Bet D, step 0: cache the members' feature frame (cluster/w2_features.assemble_w2(adsb=True, extra=True)) for
the laptop subset experiments, so each subset fit does not rebuild congestion / weather / ADS-B joins.

Writes cache/seq/w2x_training.parquet (DEP order of load('training', phase='DEP'); columns = the 7 members'
regressor columns + MVT_ID_mvt, y, month) and cache/seq/w2x_groups.json. Peak ~4.5 GB (assemble_w2).

    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 nice -n 5 \
        /home/thinklex/prc2026-venv/bin/python experiments/seq/s00_cache_frame.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from cluster.w2_features import _one, _align_ades, columns
from prc.features import align_categories
from prc.config import CACHE, TARGET, ID
from prc.io import peak_rss_gb

OUT = CACHE / 'seq'


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    X, meta, groups = _one('training', adsb=True, extra=True)
    X, = align_categories(X)
    _align_ades(X)
    cols, clf_cols = columns(groups)
    X = X[cols].copy()
    X.insert(0, ID, meta[ID].to_numpy())
    X['y'] = meta[TARGET].to_numpy().astype(float)
    X['month'] = meta['month'].to_numpy().astype('int8')
    for c in X.columns:
        if X[c].dtype == np.float64 and c not in ('y', ID):          # MVT_ID needs float64 (28 bits)
            X[c] = X[c].astype('float32')
    X.to_parquet(OUT / 'w2x_training.parquet', index=False)
    json.dump({'groups': groups, 'cols': cols, 'clf_cols': clf_cols}, open(OUT / 'w2x_groups.json', 'w'), indent=1)
    print(f'saved {X.shape} in {time.time() - t0:.0f} s | peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
