import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd, resource
D = os.path.dirname(os.path.abspath(__file__))
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache'

def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

def load_dep(cols=None, which='training'):
    return pd.read_parquet(f'{D}/{which}_DEP.parquet', columns=cols)

def preds():
    o = np.load(f'{R}/exp62_oof.npy')          # v5, v6, v6x(=v7 recipe)
    b = np.load(f'{R}/blend_oof_mix_nn3.npy')   # lgb mix 1 seed, 3-seed net mix
    v6, v7 = o[1], o[2]
    # approx v8: outside P the tail branch is inert -> average v7 with the net mixture;
    # inside P keep v7 (specialist) -- approximation, net pb not separately cached
    return v6, v7, b

def boot(y, a, b, n=500, seed=0, idx_groups=None):
    da = (y - a) ** 2; db = (y - b) ** 2
    rng = np.random.default_rng(seed); N = len(y); out = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, N, N)
        out[i] = np.sqrt(da[ix].mean()) - np.sqrt(db[ix].mean())
    return out.mean(), out.std(), np.percentile(out, 2.5), np.percentile(out, 97.5)

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

def S_(s):
    """categorical/object series -> numpy str array with 'NA' for missing"""
    return pd.Series(s).astype(object).where(pd.Series(s).notna(), 'NA').astype(str).to_numpy()

def K(*parts):
    out = np.asarray(parts[0], dtype=object)
    for p in parts[1:]:
        out = out + '|' + np.asarray(p, dtype=object)
    return out
