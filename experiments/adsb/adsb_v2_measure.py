"""Measure ADS-B v2 on the labelled local days: coverage, accuracy, and a leave-one-day-out residual screen.

    python experiments/adsb/adsb_v2_measure.py [--v2 cache/adsb_lol/features_v2_training_local.parquet] [--seeds 3]

Labels are used ONLY here (measurement + the screen's residual model), never in the features.
A. Coverage per airport: share of ALL departures with a production off-block (adsb_off_best) and with the v2 combined
   estimate (adsb2_off_best2 = production, else back-calculated).
B. Accuracy on the flights the v2 estimators add (no production off-block, BLOCK < MVT, label <= 7200 s):
   |taxi implied - label| = |BLOCK - estimate| for production first point, v2 reappearance point, back-calc from stand,
   back-calc from roll distance, AOBT_3 (NM proxy). Median, p90, and share closer than AOBT_3.
C. Residual screen on the v15 leave-one-month-out OOF (cache/w2/v15s_oof.npy): LightGBM on y - base, leave-one-DAY-out,
   fit and applied on rows with production ground points OR a v2 departure run (same rows in every arm):
     C   control: APT, base, MVT-AOBT3, MVT-SCHED
     OLD production adsb_* features (what the models already have) + control columns
     NEW OLD + adsb2_* columns
     PLB placebo: OLD + adsb2_* permuted within (day, APT) -> the gain from extra columns carrying no information
     OLDm / NEWm redundancy check: OLD without the production first-point columns (adsb_off_first, adsb_gs_first,
          adsb_d_stand_first, adsb_stand_known), without / with the adsb2_* columns. If NEWm recovers OLD, the v2
          columns carry the same information as those four.
   --mlp adds the same OLD / NEW comparison with a small MLP residual model (smooth learner, as a proxy for the
   network members, which may not learn off_first + f(distance) as easily as trees).
   Paired cluster bootstrap over (day x airport), 2000 resamples: NEW-OLD and PLB-OLD.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_match_measure import load_deps  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / 'cache'
BASE = CACHE / 'adsb_lol'
NEW_EXCLUDE = {'adsb2_best2_src'}


def q(s, p):
    s = pd.Series(s).dropna()
    return np.nan if s.empty else float(np.quantile(s, p))


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def load(v2_path):
    v2 = pd.read_parquet(v2_path)
    prod = pd.read_parquet(BASE / 'features_training.parquet')
    prod = prod[prod.MVT_ID_mvt.isin(v2.MVT_ID_mvt)]
    ok = v2.adsb2_run.notna() | v2.MVT_ID_mvt.isin(prod.MVT_ID_mvt[prod.adsb_day_ok == 1])
    v2 = v2[ok]
    dt = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    dt['base'] = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    days = sorted(dt[dt.MVT_ID_mvt.isin(v2.MVT_ID_mvt)].MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d').unique())
    dep = load_deps(days, 'training')
    m = dep.merge(prod, on='MVT_ID_mvt', how='left').merge(v2, on='MVT_ID_mvt', how='inner')
    m = m.merge(dt[['MVT_ID_mvt', 'base']], on='MVT_ID_mvt', how='left')
    if m.base.isna().any():
        raise SystemExit('ERROR: rows without base')
    m['day'] = pd.to_datetime(m.mvt_s, unit='s').dt.strftime('%Y-%m-%d')
    m['APT'] = m.APT.astype(str)
    m['y'] = m.TAXITIME_SEC_mvt.astype(float)
    m['off_aobt3'] = m.mvt_s - m.aobt3_s
    m['off_sched'] = m.mvt_s - m.sched_s
    return m


def coverage_accuracy(m):
    rows = []
    acc = (m.block_s < m.mvt_s) & (m.y <= 7200)
    for apt, g in list(m.groupby('APT')) + [('ALL', m)]:
        n = len(g)
        prod_b = g.adsb_off_best.notna()
        new = g.adsb2_off_best2.notna() & ~prod_b
        r = dict(APT=apt, n=n, prod_best_pct=100 * prod_b.mean(), best2_pct=100 * g.adsb2_off_best2.notna().mean(),
                 add_pct=100 * new.mean(), add_stand_pct=100 * (new & (g.adsb2_best2_src == 2)).mean(),
                 add_roll_pct=100 * (new & (g.adsb2_best2_src == 3)).mean(),
                 prod_ground_pct=100 * (g.adsb_has_ground == 1).mean(), run_pct=100 * (g.adsb2_run == 1).mean(),
                 contam_pct=100 * (g.adsb2_contam == 1).mean())
        a = g[acc.loc[g.index] & new & g.off_aobt3.notna()]
        r['n_acc'] = len(a)
        for name, col in (('first', 'adsb_off_first'), ('R', 'adsb2_off_R'), ('bcS', 'adsb2_off_bc_stand'),
                          ('bcR', 'adsb2_off_bc_roll'), ('best2', 'adsb2_off_best2'), ('aobt3', 'off_aobt3')):
            e = (a[col] - a.y)
            r[f'{name}_med'] = q(e.abs(), .5)
            r[f'{name}_p90'] = q(e.abs(), .9)
            r[f'{name}_bias'] = q(e, .5)
        ea = (a.adsb2_off_best2 - a.y).abs()
        eo = (a.off_aobt3 - a.y).abs()
        r['best2_closer_than_aobt3_pct'] = 100 * (ea < eo).mean() if len(a) else np.nan
        # the production estimate on its own rows, for reference
        b = g[acc.loc[g.index] & prod_b]
        r['prod_best_med'] = q((b.adsb_off_best - b.y).abs(), .5)
        r['prod_best_p90'] = q((b.adsb_off_best - b.y).abs(), .9)
        rows.append(r)
    return pd.DataFrame(rows).set_index('APT')


def contam_check(m):
    """Flights whose production segment starts > 600 s before the v2 run: which start is closer to BLOCK?"""
    acc = (m.block_s < m.mvt_s) & (m.y <= 7200)
    c = m[acc & (m.adsb2_contam == 1)]
    rows = []
    for apt, g in list(c.groupby('APT')) + [('ALL', c)]:
        rows.append(dict(APT=apt, n=len(g), prod_best_pct=100 * g.adsb_off_best.notna().mean(),
                         first_minus_block_med=q(g.adsb_off_first - g.y, .5) * -1,
                         R_minus_block_med=q(g.adsb2_off_R - g.y, .5) * -1,
                         first_abs_med=q((g.adsb_off_first - g.y).abs(), .5),
                         R_abs_med=q((g.adsb2_off_R - g.y).abs(), .5)))
    return pd.DataFrame(rows).set_index('APT')


def screen(m, seeds=3, rows='ground'):
    y = m.y.to_numpy()
    base = m.base.to_numpy()
    prod_cols = [c for c in m.columns if c.startswith('adsb_') and c not in ('adsb_day_ok',)]
    new_cols = [c for c in m.columns if c.startswith('adsb2_') and c not in NEW_EXCLUDE]
    X = m[prod_cols + new_cols].astype(float).copy()
    X['apt'] = m.APT.astype('category')
    X['base'] = base
    X['off_aobt3'] = m.off_aobt3
    X['off_sched'] = m.off_sched
    ctrl = ['apt', 'base', 'off_aobt3', 'off_sched']
    # placebo: permute the v2 block row-wise within (day, APT), keeping its joint structure
    rng = np.random.default_rng(12345)
    P = X.copy()
    grp = (m.day + '_' + m.APT).to_numpy()
    perm = np.arange(len(m))
    for gk in np.unique(grp):
        ix = np.flatnonzero(grp == gk)
        perm[ix] = rng.permutation(ix)
    P[new_cols] = X[new_cols].to_numpy()[perm]
    use = ((m.adsb_has_ground == 1) | (m.adsb2_run == 1)).to_numpy()
    if rows == 'aptday':        # every departure of an airport-day with ADS-B ground coverage (congestion applies to all)
        use = (m.adsb_apt_day_ground.fillna(0) > 0).to_numpy()
    days = sorted(m.day.unique())
    first_cols = ['adsb_off_first', 'adsb_gs_first', 'adsb_d_stand_first', 'adsb_stand_known']
    prod_m = [c for c in prod_cols if c not in first_cols]
    cong_cols = [c for c in new_cols if c.startswith('adsb2_cong')]
    est_cols = [c for c in new_cols if c not in cong_cols]
    arms = {'C': (X, ctrl), 'OLD': (X, ctrl + prod_cols), 'NEW': (X, ctrl + prod_cols + new_cols),
            'PLB': (P, ctrl + prod_cols + new_cols), 'OLDm': (X, ctrl + prod_m), 'NEWm': (X, ctrl + prod_m + est_cols)}
    if cong_cols:
        arms['EST'] = (X, ctrl + prod_cols + est_cols)
        arms['CONG'] = (X, ctrl + prod_cols + cong_cols)
    pred = {k: base.copy() for k in arms}
    for d in days:
        te = (m.day == d).to_numpy()
        tr = ~te & (y <= 7200) & use
        ap_ = te & use
        for k, (XX, cols) in arms.items():
            acc = np.zeros(ap_.sum())
            for s in range(seeds):
                mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                        verbose=-1, random_state=s)
                mdl.fit(XX.loc[tr, cols], y[tr] - base[tr])
                acc += mdl.predict(XX.loc[ap_, cols])
            pred[k][ap_] = np.clip(base[ap_] + acc / seeds, 0, None)
    out_rows = []
    for apt, g in list(m.groupby('APT')) + [('ALL', m)]:
        ix = g.index.to_numpy()
        r = dict(APT=apt, n=len(ix), use_pct=100 * use[ix].mean(), rmse_base=rmse(y[ix] - base[ix]))
        for k in arms:
            r[f'rmse_{k}'] = rmse(y[ix] - pred[k][ix])
        r['OLD_minus_base'] = r['rmse_OLD'] - r['rmse_base']
        r['NEW_minus_OLD'] = r['rmse_NEW'] - r['rmse_OLD']
        r['PLB_minus_OLD'] = r['rmse_PLB'] - r['rmse_OLD']
        r['C_minus_base'] = r['rmse_C'] - r['rmse_base']
        r['OLDm_minus_OLD'] = r['rmse_OLDm'] - r['rmse_OLD']
        r['NEWm_minus_OLD'] = r['rmse_NEWm'] - r['rmse_OLD']
        for k in ('EST', 'CONG'):
            if k in pred:
                r[f'{k}_minus_OLD'] = r[f'rmse_{k}'] - r['rmse_OLD']
        out_rows.append(r)
    res = pd.DataFrame(out_rows).set_index('APT')
    return res, *_boot(m, y, pred), pred


def _boot(m, y, pred):
    cl = (m.day + '_' + m.APT).to_numpy()
    uc = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in uc}
    sse = {k: {c: float(np.sum((y[i] - p[i]) ** 2)) for c, i in idx.items()} for k, p in pred.items()}
    nn = {c: len(i) for c, i in idx.items()}
    rs = np.random.default_rng(0)
    pairs = [(a, b) for a, b in (('NEW', 'OLD'), ('PLB', 'OLD'), ('OLD', 'C'), ('OLDm', 'OLD'), ('NEWm', 'OLD'),
                                 ('NEWm', 'OLDm'), ('EST', 'OLD'), ('CONG', 'OLD'), ('NEW', 'PLB'))
             if a in pred and b in pred]
    boot = {f'{a}-{b}': [] for a, b in pairs}
    for _ in range(2000):
        s = rs.choice(uc, len(uc))
        N = sum(nn[c] for c in s)
        r_ = {k: np.sqrt(sum(sse[k][c] for c in s) / N) for k in pred}
        for a, b in pairs:
            boot[f'{a}-{b}'].append(r_[a] - r_[b])
    ci = {k: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) for k, v in boot.items()}
    per_day = pd.DataFrame({'day': m.day, **{k: (y - p) ** 2 for k, p in pred.items()}}).groupby('day').mean() ** 0.5
    for a, b in pairs:
        per_day[f'{a}-{b}'] = per_day[a] - per_day[b]
    return ci, per_day


def screen_mlp(m, seeds=2):
    """OLD vs NEW with a small MLP residual model (quantile-normalised inputs, NaN -> 0 + missing indicators)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import QuantileTransformer
    y = m.y.to_numpy()
    base = m.base.to_numpy()
    prod_cols = [c for c in m.columns if c.startswith('adsb_') and c not in ('adsb_day_ok',)]
    new_cols = [c for c in m.columns if c.startswith('adsb2_') and c not in NEW_EXCLUDE]
    num = m[prod_cols + new_cols + ['off_aobt3', 'off_sched']].astype(float).copy()
    num['base'] = base
    apt = pd.get_dummies(m.APT).astype(float)
    use = ((m.adsb_has_ground == 1) | (m.adsb2_run == 1)).to_numpy()
    days = sorted(m.day.unique())
    arms = {'OLD': prod_cols + ['off_aobt3', 'off_sched', 'base'], 'NEW': prod_cols + new_cols + ['off_aobt3', 'off_sched', 'base']}
    pred = {k: base.copy() for k in arms}
    for d in days:
        te = (m.day == d).to_numpy()
        tr = ~te & (y <= 7200) & use
        ap_ = te & use
        for k, cols in arms.items():
            qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=100000, random_state=0)
            A = num[cols].to_numpy()
            qt.fit(A[tr])
            Z = np.nan_to_num(qt.transform(A), nan=0.0)
            Z = np.c_[Z, np.isnan(A).astype(float), apt.to_numpy()]
            tgt = np.clip(y - base, -1500, 1500)
            acc = np.zeros(ap_.sum())
            for s in range(seeds):
                mdl = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=60,
                                   early_stopping=True, validation_fraction=0.15, n_iter_no_change=6, random_state=s)
                mdl.fit(Z[tr], tgt[tr] / 300.0)
                acc += mdl.predict(Z[ap_]) * 300.0
            pred[k][ap_] = np.clip(base[ap_] + acc / seeds, 0, None)
    rows = []
    for a_, g in list(m.groupby('APT')) + [('ALL', m)]:
        ix = g.index.to_numpy()
        r = dict(APT=a_, rmse_base=rmse(y[ix] - base[ix]), rmse_OLD=rmse(y[ix] - pred['OLD'][ix]),
                 rmse_NEW=rmse(y[ix] - pred['NEW'][ix]))
        r['NEW_minus_OLD'] = r['rmse_NEW'] - r['rmse_OLD']
        rows.append(r)
    ci, per_day = _boot(m, y, pred)
    return pd.DataFrame(rows).set_index('APT'), ci, per_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', type=Path, default=BASE / 'features_v2_training_local.parquet')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', type=Path, default=BASE / 'match' / 'v2_measure')
    ap.add_argument('--mlp', action='store_true')
    ap.add_argument('--rows', default='ground', choices=['ground', 'aptday'])
    a = ap.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    m = load(a.v2)
    print(f'{len(m):,} departures on {m.day.nunique()} labelled days: {sorted(m.day.unique())}')
    ca = coverage_accuracy(m)
    print('\n== A/B coverage (% of all departures) and accuracy on the ADDED flights (|BLOCK - estimate| s)')
    print(ca.round(1).to_string())
    cc = contam_check(m)
    print('\n== production segment starting > 600 s before the v2 run (adsb2_contam)')
    print(cc.round(1).to_string())
    res, ci, per_day, pred = screen(m, a.seeds, a.rows)
    print('\n== C residual screen (RMSE, s)')
    print(res.round(2).to_string())
    for k, (lo, hi) in ci.items():
        print(f'ALL {k}: 95% CI [{lo:+.2f}, {hi:+.2f}]')
    print(per_day.round(2).to_string())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ca.to_csv(f'{a.out}_coverage_accuracy.csv')
    cc.to_csv(f'{a.out}_contam.csv')
    res.to_csv(f'{a.out}_screen.csv')
    per_day.to_csv(f'{a.out}_screen_days.csv')
    pd.DataFrame([(k, res.loc['ALL', f'rmse_{k.split("-")[0]}'] - res.loc['ALL', f'rmse_{k.split("-")[1]}'], lo, hi)
                  for k, (lo, hi) in ci.items()], columns=['pair', 'delta', 'ci_lo', 'ci_hi']).to_csv(
        f'{a.out}_screen_ci.csv', index=False)
    if a.mlp:
        r2, ci2, pd2 = screen_mlp(m)
        print('\n== MLP residual screen (RMSE, s)')
        print(r2.round(2).to_string())
        for k, (lo, hi) in ci2.items():
            print(f'ALL MLP {k}: 95% CI [{lo:+.2f}, {hi:+.2f}]')
        print(pd2.round(2).to_string())
        r2.to_csv(f'{a.out}_screen_mlp.csv')
    pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt, 'y': m.y, **{f'pred_{k}': v for k, v in pred.items()}}).to_parquet(
        f'{a.out}_preds.parquet', index=False)


if __name__ == '__main__':
    main()
