"""How much would adsb.lol features lower OUR model's RMSE?  (sample days only)

    python experiments/adsb/adsb_gain_estimate.py cache/adsb_lol/match/training_<a>_<b>.parquet

Base = the v15-structure leave-one-month-out OOF prediction (cache/w2/v15s_oof.npy, aligned
with cache/major/dep_training.parquet by row order), i.e. what the shipped model predicts for a
July 2025 flight when July was held out.

Correction models, all fit leave-one-DAY-out over the matched sample days and evaluated on
ALL departures of those days (flights without ADS-B keep the base prediction):
  L  per-airport linear stack on rows with an ADS-B off-block (t_best):
       y ~ a + b * base + c * (MVT - t_best)
  G  LightGBM on the residual y - base with ADS-B features (+ APT, base, MVT-AOBT3, MVT-SCHED), fit and
     applied only on departures with ADS-B ground points; all others keep the base. Small, heavily
     regularised (the sample is a few days).
  C  CONTROL: the same LightGBM on the same rows with only APT, base, MVT-AOBT3, MVT-SCHED (no ADS-B),
     to separate in-month recalibration from the ADS-B information.
Reported: RMSE of base / L / G / C per airport and overall, the delta, and a paired bootstrap
95% CI over DAYS x AIRPORT clusters (rows are not independent within a day).
Caveat: few days -> the estimate is noisy; it is a feasibility number, not a CV result.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / 'cache'


def features(m):
    f = pd.DataFrame(index=m.index)
    mv = m.mvt_s
    for c in ('t_best', 't_first', 't_move', 't_stat_end', 't_push_prev', 't_roll', 't_last', 't_to'):
        f['off_' + c] = mv - m['ev_' + c]
    for c in ('stat_s', 'move_gap_s', 'gs_first', 'n_pts', 'max_gap_s', 'gap_last_to_air', 'path_m',
              'n_stops', 'stop_s', 'dwell_s', 'push_m', 'n_push', 'mlat_frac', 'cov_s_per_pt'):
        f[c] = m['ev_' + c].astype(float)
    f['hold_s'] = m.ev_t_to - m.ev_t_last_stop_end
    f['at_park'] = m.ev_at_park.astype(float)
    f['best_src'] = m.ev_best_src.map({'push': 2, 'rest': 1, 'none': 0}).astype(float)
    f['to_obs'] = (m.ev_to_src == 'obs').astype(float).where(m.match != 'none')
    f['matched'] = (m.match != 'none').astype(float)
    f['apt'] = m.APT.astype(str).astype('category')
    p = m[(m.ev_best_src == 'push') & ((m.ev_t_push_prev - m.ev_t_first) < 600)]
    st = p.groupby(['APT', 'STAND_mvt'], observed=True)[['ev_x_first', 'ev_y_first']].median()
    st = st[p.groupby(['APT', 'STAND_mvt'], observed=True).size() >= 3]
    j = m[['APT', 'STAND_mvt']].join(st, on=['APT', 'STAND_mvt'])
    f['d_stand_first'] = np.hypot(m.ev_x_first - j.ev_x_first, m.ev_y_first - j.ev_y_first)
    f['base'] = m.base
    f['base_minus_best'] = m.base - f.off_t_best
    f['off_aobt3'] = mv - m.aobt3_s
    f['off_sched'] = mv - m.sched_s
    return f


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    mp = Path(sys.argv[1])
    m = pd.read_parquet(mp)
    dt = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt'])
    dt['base'] = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    m = m.merge(dt, on='MVT_ID_mvt', how='left')
    if m.base.isna().any():
        raise SystemExit(f'ERROR: {m.base.isna().sum()} rows without a v15 OOF prediction')
    m['day'] = pd.to_datetime(m.mvt_s, unit='s').dt.strftime('%Y-%m-%d')
    y = m.TAXITIME_SEC_mvt.astype(float).to_numpy()
    base = m.base.to_numpy()
    F = features(m)
    days = sorted(m.day.unique())
    print(f'{len(m):,} departures on {len(days)} days: {days}')

    predL = base.copy()
    predG = base.copy()
    predC = base.copy()          # CONTROL: same model and rows, but NO ADS-B features
    ctrl_cols = ['apt', 'base', 'off_aobt3', 'off_sched']
    has = F.off_t_best.notna().to_numpy()
    gnd = (m.ev_n_seg.fillna(0) > 0).to_numpy()
    for d in days:
        te = (m.day == d).to_numpy()
        tr = ~te
        # L: per airport linear stack on rows with t_best (fallback: pooled)
        for apt in m.APT.astype(str).unique():
            a = (m.APT.astype(str) == apt).to_numpy()
            trr, ter = tr & a & has, te & a & has
            if ter.sum() == 0:
                continue
            if trr.sum() < 30:
                trr = tr & has
            X = np.c_[np.ones(trr.sum()), base[trr], F.off_t_best.to_numpy()[trr]]
            # fit only on non-tail rows so a handful of tails do not dominate 30-row fits
            keep = y[trr] <= 7200
            coef, *_ = np.linalg.lstsq(X[keep], y[trr][keep], rcond=None)
            Xt = np.c_[np.ones(ter.sum()), base[ter], F.off_t_best.to_numpy()[ter]]
            predL[ter] = np.clip(Xt @ coef, 0, None)
        # G: LightGBM residual model, fit and applied ONLY where ADS-B saw the aircraft on the ground
        #    (elsewhere the base prediction is kept, so airports without coverage cannot get worse)
        keep = tr & (y <= 7200) & gnd
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                verbose=-1, random_state=0)
        mdl.fit(F[keep], y[keep] - base[keep])
        ap_ = te & gnd
        predG[ap_] = np.clip(base[ap_] + mdl.predict(F[ap_]), 0, None)
        ctl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                verbose=-1, random_state=0)
        ctl.fit(F.loc[keep, ctrl_cols], y[keep] - base[keep])
        predC[ap_] = np.clip(base[ap_] + ctl.predict(F.loc[ap_, ctrl_cols]), 0, None)

    rows = []
    for apt, g in list(m.groupby('APT', observed=True)) + [('ALL', m)]:
        ix = g.index.to_numpy()
        rows.append(dict(APT=apt, n=len(ix), pct_best=100 * has[ix].mean(),
                         rmse_base=rmse(y[ix] - base[ix]), rmse_L=rmse(y[ix] - predL[ix]),
                         rmse_G=rmse(y[ix] - predG[ix]), rmse_C=rmse(y[ix] - predC[ix]),
                         rmse_base_on_best=rmse(y[ix][has[ix]] - base[ix][has[ix]]) if has[ix].any() else np.nan,
                         rmse_L_on_best=rmse(y[ix][has[ix]] - predL[ix][has[ix]]) if has[ix].any() else np.nan,
                         rmse_G_on_best=rmse(y[ix][has[ix]] - predG[ix][has[ix]]) if has[ix].any() else np.nan))
    r = pd.DataFrame(rows).set_index('APT')
    r['dL'] = r.rmse_L - r.rmse_base
    r['dG'] = r.rmse_G - r.rmse_base
    r['dC'] = r.rmse_C - r.rmse_base
    r['dG_vs_C'] = r.rmse_G - r.rmse_C
    pd.set_option('display.width', 250)
    print(r.round(2).to_string())

    # paired cluster bootstrap over (day, airport)
    cl = (m.day + '_' + m.APT.astype(str)).to_numpy()
    uc = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in uc}
    rng = np.random.default_rng(0)
    sse = {k: {c: float(np.sum((y[i] - p[i]) ** 2)) for c, i in idx.items()}
           for k, p in (('b', base), ('L', predL), ('G', predG), ('C', predC))}
    nn = {c: len(i) for c, i in idx.items()}
    dl, dg, dc, dgc = [], [], [], []
    for _ in range(2000):
        s = rng.choice(uc, len(uc))
        N = sum(nn[c] for c in s)
        rb = np.sqrt(sum(sse['b'][c] for c in s) / N)
        rc = np.sqrt(sum(sse['C'][c] for c in s) / N)
        dl.append(np.sqrt(sum(sse['L'][c] for c in s) / N) - rb)
        dg.append(np.sqrt(sum(sse['G'][c] for c in s) / N) - rb)
        dc.append(rc - rb)
        dgc.append(np.sqrt(sum(sse['G'][c] for c in s) / N) - rc)
    print(f"ALL-flight delta L {r.loc['ALL', 'dL']:+.2f} s  95% CI [{np.quantile(dl, .025):+.2f}, {np.quantile(dl, .975):+.2f}]")
    print(f"ALL-flight delta G {r.loc['ALL', 'dG']:+.2f} s  95% CI [{np.quantile(dg, .025):+.2f}, {np.quantile(dg, .975):+.2f}]")
    print(f"ALL-flight delta C (control, no ADS-B) {r.loc['ALL', 'dC']:+.2f} s  95% CI [{np.quantile(dc, .025):+.2f}, {np.quantile(dc, .975):+.2f}]")
    print(f"ALL-flight G - C (ADS-B contribution) {r.loc['ALL', 'dG_vs_C']:+.2f} s  95% CI [{np.quantile(dgc, .025):+.2f}, {np.quantile(dgc, .975):+.2f}]")
    # same, excluding the > 2 h tail rows (where neither can help much)
    t = y <= 7200
    print(f"rows y<=7200: base {rmse(y[t] - base[t]):.2f}  L {rmse(y[t] - predL[t]):.2f}  G {rmse(y[t] - predG[t]):.2f}  C {rmse(y[t] - predC[t]):.2f}")
    m['_d'] = m.day
    per_day = pd.DataFrame({'day': m.day, 'e_b': (y - base) ** 2, 'e_g': (y - predG) ** 2, 'e_c': (y - predC) ** 2}).groupby('day').mean() ** 0.5
    per_day['dG'] = per_day.e_g - per_day.e_b
    per_day['dC'] = per_day.e_c - per_day.e_b
    print(per_day.round(2).to_string())
    r.to_csv(mp.with_suffix('.gain.csv'))
    out = mp.with_suffix('.gain_preds.parquet')
    pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt, 'base': base, 'predL': predL, 'predG': predG, 'predC': predC,
                  'y': y}).to_parquet(out)


if __name__ == '__main__':
    main()
