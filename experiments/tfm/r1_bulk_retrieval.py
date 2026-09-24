"""R1 -- does a tabular foundation model add anything in the BULK when it gets the right context?

s04 (2026-09-24) gave TabPFN a RANDOM 2,000-row context and asked it to predict taxi time
directly: it beat LightGBM on the same 2,000 rows but lost to LightGBM on all rows, and its blend
weight on top of production was 0. Three things were left untested, and this script tests them:
  - TabICL, which is built for large contexts, with 10,000 context rows instead of 2,000;
  - RETRIEVAL: for each group of test flights, the 10,000 most similar flights of the other
    months as context, instead of a random sample;
  - predicting the RESIDUAL of our current model (v11 stand-in, cache/strategy/v11s_oof.npy)
    with that model's prediction as an extra feature, so the question is exactly "does it add
    signal on top of what we ship".

Cells (airport, held-out month): LIRF 7, EDDF 1, EHAM 7, LFPG 1 -- two scored months, two of the
biggest airports. Domain = departures outside P (no label-based filtering of test rows).
Test = N_TEST random flights of the held-out month; context pool = the airport's other 11 months.

Arms, all predicting r = y - v11~ (context targets clipped to +-3600 s against artefact tails):
  lgb     CONTROL: LightGBM on the whole pool, same features. If the foundation model gains no
          more than this, it is only acting as a stacker, not bringing anything new.
  rand    TabICL regressor, context = CTX random pool rows
  knn     TabICL regressor, per cluster of ~100 test flights, context = the CTX pool rows closest
          to any member, in a space of the 12 most important features (LightGBM gain on the pool,
          quantile-normalised, weighted by sqrt(gain share))
Scoring: served = v11~ + w * r_hat. w is chosen on the even test rows (grid 0..1) and scored on
the odd rows; w = 0.5 (fixed in advance) gets a paired bootstrap on all test rows.

PRE-REGISTERED KILL: if `knn` and `rand` fail to beat v11~ by >= 0.5% RMSE on the odd rows in at
least 3 of the 4 cells, or fail to beat the `lgb` control, the bulk foundation-model idea is dead.

Resumable: one npz per cell under cache/tfm_r1/. GPU recommended (cluster/r1_array.sbatch);
laptop only for a smoke test:  --cells LIRF:7 --n-test 200 --ctx 1000
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, lightgbm as lgb
from prc.config import CACHE
from prc.validation import paired_bootstrap
from prc.io import peak_rss_gb

OUT = CACHE / 'tfm_r1'
CLIP = 3600.0
N_KEY = 12
CLUSTER = 100


def load():
    X = pd.read_parquet(CACHE / 'altfam_X.parquet')
    y = np.load(CACHE / 'altfam_y.npy').astype(float)
    mon = np.load(CACHE / 'altfam_month.npy')
    base = np.load(CACHE / 'strategy' / 'v11s_oof.npy').astype(float)
    if not (len(X) == len(y) == len(mon) == len(base)):
        sys.exit('FATAL: altfam_X / altfam_y / altfam_month / v11s_oof lengths differ')
    oa, so = X.off_aobt3.to_numpy(), X.off_sched.to_numpy()
    P = ~np.isfinite(oa) & (so > 7200)
    apt = X.APT.astype(str).to_numpy()
    X = X.drop(columns=['APT'])
    for c in X.columns:
        if str(X[c].dtype) == 'category':
            X[c] = X[c].cat.codes.replace(-1, np.nan).astype('float32')
    X['v11_pred'] = base.astype('float32')
    return X.astype('float32'), y, mon, base, P, apt


def tabicl_fit_predict(Xc, rc, Xq, device, threads, batch_size, seed):
    from tabicl import TabICLRegressor
    reg = TabICLRegressor(device=device, random_state=seed, n_jobs=threads, batch_size=batch_size)
    reg.fit(Xc, rc)
    return np.concatenate([reg.predict(Xq[i:i + 1000]) for i in range(0, len(Xq), 1000)])


def quantile_space(Xpool, Xtest, cols, weights):
    """Map each key feature to N(0,1) by the pool's empirical quantiles; NaN -> 0 (the median)."""
    Zp, Zt = np.zeros((len(Xpool), len(cols)), 'float32'), np.zeros((len(Xtest), len(cols)), 'float32')
    from scipy.stats import norm
    for j, c in enumerate(cols):
        v = Xpool[:, c]; ok = np.isfinite(v)
        qs = np.unique(np.quantile(v[ok], np.linspace(0, 1, 201))) if ok.any() else np.array([0.0])
        for src, dst in ((Xpool, Zp), (Xtest, Zt)):
            u = src[:, c]; f = np.isfinite(u)
            pr = np.interp(u[f], qs, np.linspace(0, 1, len(qs))) if len(qs) > 1 else np.full(f.sum(), .5)
            dst[f, j] = norm.ppf(np.clip(pr, 0.005, 0.995)) * weights[j]
    return Zp, Zt


def run_cell(X, y, mon, base, P, apt, A, m, a):
    f = OUT / f'r1_{A}_m{m:02d}_n{a.n_test}_c{a.ctx}.npz'
    if f.exists():
        print(f'{A} {m}: done already ({f.name})'); return
    rng = np.random.default_rng(0)
    dom = (apt == A) & ~P
    pool = np.flatnonzero(dom & (mon != m))
    te = np.sort(rng.choice(np.flatnonzero(dom & (mon == m)), a.n_test, replace=False))
    r = y - base
    Xp, Xt = X.iloc[pool].to_numpy(), X.iloc[te].to_numpy()
    rp = np.clip(r[pool], -CLIP, CLIP)
    out = dict(te=te, y=y[te], base=base[te])
    t0 = time.time()

    # control: LightGBM residual model on the whole pool
    gbm = lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=a.threads,
                         verbose=-1, seed=0), lgb.Dataset(Xp, rp), 400)
    out['lgb'] = gbm.predict(Xt)
    gain = gbm.feature_importance('gain'); key = np.argsort(-gain)[:N_KEY]
    w = np.sqrt(gain[key] / gain[key].sum())
    print(f'{A} {m}: pool {len(pool):,}, test {len(te)}; lgb control {time.time() - t0:.0f} s; key features '
          + ', '.join(X.columns[k] for k in key[:6]) + ' ...', flush=True)

    # rand: one random context
    sel = np.sort(rng.choice(len(pool), min(a.ctx, len(pool)), replace=False))
    t1 = time.time()
    out['rand'] = tabicl_fit_predict(Xp[sel], rp[sel], Xt, a.device, a.threads, a.batch_size, 0)
    print(f'{A} {m}: rand {time.time() - t1:.0f} s', flush=True)

    # knn: retrieval context per cluster of test flights
    Zp, Zt = quantile_space(Xp, Xt, key, w)
    from sklearn.cluster import KMeans
    k = max(1, len(te) // CLUSTER)
    lab = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(Zt)
    pk = np.full(len(te), np.nan)
    t1 = time.time()
    for c in range(k):
        mem = np.flatnonzero(lab == c)
        dmin = np.full(len(Zp), np.inf, 'float32')
        for i in range(0, len(mem), 25):
            dd = ((Zp[:, None, :] - Zt[mem[i:i + 25]][None, :, :]) ** 2).sum(-1).min(1)
            dmin = np.minimum(dmin, dd)
        ctx = np.argpartition(dmin, min(a.ctx, len(dmin) - 1))[:a.ctx]
        pk[mem] = tabicl_fit_predict(Xp[ctx], rp[ctx], Xt[mem], a.device, a.threads, a.batch_size, 0)
    out['knn'] = pk
    print(f'{A} {m}: knn {k} clusters {time.time() - t1:.0f} s | peak {peak_rss_gb():.2f} GB', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)


def report(cells, a):
    print(f'\nR1 report: served = v11~ + w * r_hat; w fit on even test rows, scored on odd rows')
    print(f"{'cell':10s}{'n':>6s}{'v11~ odd':>10s}" + ''.join(f'{x + " w*":>9s}{x + " odd":>10s}{x + " %":>8s}' for x in ('lgb', 'rand', 'knn'))
          + f"{'corr knn,lgb':>13s}")
    boot = []
    for A, m in cells:
        f = OUT / f'r1_{A}_m{m:02d}_n{a.n_test}_c{a.ctx}.npz'
        if not f.exists():
            print(f'{A}:{m} missing'); continue
        z = np.load(f); yt, b = z['y'], z['base']
        h = np.arange(len(yt)) % 2 == 0
        rm = lambda p, k: float(np.sqrt(np.mean((yt[k] - p[k]) ** 2)))
        row = f'{A}:{m:<5d}{len(yt):6d}{rm(b, ~h):10.1f}'
        for x in ('lgb', 'rand', 'knn'):
            ws = np.linspace(0, 1, 21)
            wst = ws[np.argmin([rm(b + wi * z[x], h) for wi in ws])]
            o = rm(b + wst * z[x], ~h)
            row += f'{wst:9.2f}{o:10.1f}{100 * (o / rm(b, ~h) - 1):+8.2f}'
            pb = paired_bootstrap(yt, b + 0.5 * z[x], b)
            boot.append(f'  {A}:{m} {x:5s} w=0.5 all rows: {pb["diff"]:+.2f} s [{pb["lo"]:+.2f},{pb["hi"]:+.2f}]')
        row += f'{np.corrcoef(z["knn"], z["lgb"])[0, 1]:13.3f}'
        print(row)
    print('\n'.join(boot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['LIRF:7', 'EDDF:1', 'EHAM:7', 'LFPG:1'])
    ap.add_argument('--n-test', type=int, default=3000)
    ap.add_argument('--ctx', type=int, default=10000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    cells = [(c.split(':')[0], int(c.split(':')[1])) for c in a.cells]
    import torch
    torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('FATAL: --device cuda but no GPU is visible (on -p gpu, --gres=gpu:1 is mandatory)')
    if not a.report:
        X, y, mon, base, P, apt = load()
        print(f'loaded {len(y):,} rows, {X.shape[1]} features (incl. v11_pred) | peak {peak_rss_gb():.2f} GB', flush=True)
        for A, m in cells:
            run_cell(X, y, mon, base, P, apt, A, m, a)
    report(cells, a)


if __name__ == '__main__':
    main()
