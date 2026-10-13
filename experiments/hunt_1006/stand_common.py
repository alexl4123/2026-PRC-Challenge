"""Shared evaluation for the stand-bound hunt."""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'


def load_2025():
    c = pd.read_parquet(H + 'stand_ctx_training.parquet')
    z = np.load(R + 'cache/w2/v28s_oof.npz')
    o = pd.read_parquet(R + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
    assert np.array_equal(o.MVT_ID_mvt.to_numpy(), c.id.to_numpy())
    c['y'] = z['y']; c['f'] = z['f']; c['month'] = z['month']; c['P'] = z['P'].astype(bool); c['D'] = z['D'].astype(bool)
    assert np.allclose(c.y, c['T'] - c.blk)
    return c


def load_2026():
    c = pd.read_parquet(H + 'stand_ctx_ranking.parquet')
    s = pd.read_parquet(R + 'outputs/understanding-accordion_v28.parquet').set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
    c['f'] = s.reindex(c.id).to_numpy().astype(float)
    assert np.isfinite(c.f).all()
    c['month'] = pd.to_datetime(c['T'], unit='s').dt.month.to_numpy()
    return c


def rmse(e, m):
    return np.sqrt(np.mean(e[m] ** 2))


def evaluate(c, lo, hi, name, verbose=True):
    """q = clip(f, lo, hi) with NaN = no bound. Prints exactness (y outside bound) and the OOF effect."""
    y, f = c.y.to_numpy(), c.f.to_numpy()
    L = np.where(np.isfinite(lo), lo, -np.inf); U = np.where(np.isfinite(hi), hi, np.inf)
    dom = np.isfinite(lo) | np.isfinite(hi)
    exc = dom & ((y < L) | (y > U))
    q = np.minimum(np.maximum(f, L), U)
    ch = q != f
    worse = ch & (np.abs(q - y) > np.abs(f - y))
    month = c.month.to_numpy()
    held, S = ~np.isin(month, [3, 6, 9, 12]), np.isin(month, [1, 7])
    d_held = rmse(q - y, held) - rmse(f - y, held); d_S = rmse(q - y, S) - rmse(f - y, S)
    d_all = rmse(q - y, np.ones(len(y), bool)) - rmse(f - y, np.ones(len(y), bool))
    dsse = (q - y) ** 2 - (f - y) ** 2
    per_m = pd.Series(dsse).groupby(month).sum()
    if verbose:
        print(f'[{name}] domain {dom.sum():,} | EXCEPTIONS {exc.sum():,} | changed {ch.sum():,} worse {worse.sum():,} '
              f'| dRMSE all {d_all:+.4f} held {d_held:+.4f} JanJul {d_S:+.4f} | months better {int((per_m < 0).sum())}/'
              f'{int((per_m != 0).sum())} | top-row share of gain {(-dsse.min() / max(-dsse.sum(), 1e-9)) if ch.any() else 0:.2f}')
    return dict(dom=int(dom.sum()), exc=int(exc.sum()), changed=int(ch.sum()), worse=int(worse.sum()), d_all=d_all,
                d_held=d_held, d_S=d_S, q=q, exc_mask=exc, ch_mask=ch)


def apply_2026(c, lo, hi, name, rmse_live=235.3996, n=344841):
    f = c.f.to_numpy()
    L = np.where(np.isfinite(lo), lo, -np.inf); U = np.where(np.isfinite(hi), hi, np.inf)
    q = np.minimum(np.maximum(f, L), U)
    ch = q != f
    mv = q - f
    bound_sse = np.sum(mv ** 2)       # guaranteed SSE reduction if the rule holds (projection onto a convex set)
    new = np.sqrt(max(rmse_live ** 2 - bound_sse / n, 0))
    print(f'[{name} 2026] domain {(np.isfinite(lo) | np.isfinite(hi)).sum():,} | rows moved {ch.sum():,} (up {int((mv > 0).sum())}, '
          f'down {int((mv < 0).sum())}) | sum|move| {np.abs(mv).sum():,.0f} | max|move| {np.abs(mv).max() if ch.any() else 0:,.0f} '
          f'| guaranteed RMSE <= {new:.3f} ({new - rmse_live:+.3f}) if the rule holds')
    return q, ch
