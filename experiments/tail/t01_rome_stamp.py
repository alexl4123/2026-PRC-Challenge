"""Rome's stamping regime: do label-free ARRIVAL stamps tell which late departures get their off-block stamped at
the schedule? (Exploratory MEASUREMENT, not a shipping decision.)

Why: at Rome, departures whose off-block was recorded AT the scheduled time carry 42% of the route A/B/C error
(class schedB, RMSE 676 s) and are the whole zone-P tail problem (y = off_sched). Arrivals' in-block times are provided
in the ranking file too, and Rome stamps 11-15% of arrivals at SIBT (4-5% elsewhere); the daily share of stamped
arrivals correlates +0.30 with the daily share of stamped late departures (2025, measured 2026-09-27).

Candidates: Rome departures taking off > 20 min after schedule (off_sched > 1200 s), all routes incl. zone P.
Target:     stamp = |BLOCK - SCHED| <= 120 s  (the label was written at the schedule).
Arms (LightGBM binary, leave-one-month-out, 2 seeds):
  base    the members' columns (cluster/w2_features.assemble_w2(adsb=True, extra=True))
  regime  base + label-free arrival-stamp regime features:
            arr_st_1h / arr_n_1h   share / count of Rome arrivals stamped (|in-block - SIBT| <= 60 s) with in-block
                                   within +-60 min of our scheduled off-block
            arr_st_3h, arr_st_day  the same within +-3 h / on the same UTC date
            arr_st_stand_day       the same at OUR stand on the same UTC date (+ count)
            arr_st_op_day          arrivals of OUR operator on the same UTC date (+ count)
Reported: AUC and Brier per arm (pooled OOF), per route (A/B/C vs P), and per seed.

    python experiments/tail/t01_rome_stamp.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from prc.config import CACHE, ID
from prc.io import load, peak_rss_gb

OUT = CACHE / 'tail'


def regime(which):
    """Label-free (for departures) arrival-stamp regime features for every Rome departure of `which`."""
    d = load(which, columns=[ID, 'PHASE_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                             'STAND_mvt', 'AIRCRAFT_OPERATOR_flt'])
    d = d[d.APT.astype(str).eq('LIRF')].reset_index(drop=True)
    arr = d[d.PHASE_mvt.eq('ARR')]; dep = d[d.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    sec = lambda s: (s - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()
    ta = sec(arr.BLOCK_TIME_UTC_mvt); st = (np.abs(ta - sec(arr.SCHED_TIME_UTC_mvt)) <= 60).astype(float)
    o = np.argsort(ta); ta, st = ta[o], st[o]; cs = np.r_[0, np.cumsum(st)]
    td = sec(dep.SCHED_TIME_UTC_mvt)
    F = pd.DataFrame({ID: dep[ID].to_numpy()})
    for nm, w in (('1h', 3600), ('3h', 10800)):
        lo, hi = np.searchsorted(ta, td - w), np.searchsorted(ta, td + w)
        n = hi - lo; F[f'arr_n_{nm}'] = n; F[f'arr_st_{nm}'] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    aday = arr.BLOCK_TIME_UTC_mvt.dt.strftime('%Y%m%d').to_numpy(); dday = dep.SCHED_TIME_UTC_mvt.dt.strftime('%Y%m%d').to_numpy()
    ast = (np.abs(sec(arr.BLOCK_TIME_UTC_mvt) - sec(arr.SCHED_TIME_UTC_mvt)) <= 60).astype(float)
    A = pd.DataFrame({'day': aday, 'stand': arr.STAND_mvt.astype(str).to_numpy(), 'op': arr.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy(), 's': ast})
    for nm, keys in (('day', ['day']), ('stand_day', ['stand', 'day']), ('op_day', ['op', 'day'])):
        g = A.groupby(keys).s.agg(['mean', 'size'])
        Dk = pd.DataFrame({'day': dday, 'stand': dep.STAND_mvt.astype(str).to_numpy(), 'op': dep.AIRCRAFT_OPERATOR_flt.astype(str).to_numpy()})
        j = Dk.join(g, on=keys)
        F[f'arr_st_{nm}'] = j['mean'].to_numpy()
        if nm != 'day':
            F[f'arr_n_{nm}'] = j['size'].fillna(0).to_numpy()
    return F


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    from cluster.w2_features import assemble_w2, columns, W2_CATS
    X, y, month, groups = assemble_w2(adsb=True, extra=True)
    cols, _ = columns(groups)
    ids = pd.read_parquet(CACHE / 'v11' / 'feat_training.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    so = X['off_sched'].to_numpy(); rome = X['APT'].astype(str).to_numpy() == 'LIRF'
    b = np.load(CACHE / 'v11' / 'base.npz'); P = b['P']
    cand = rome & (so > 1200) & np.isfinite(y)
    blk = so - y          # MVT - SCHED - (MVT - BLOCK) = BLOCK - SCHED
    tgt = (np.abs(blk) <= 120).astype(int)
    R = regime('training').set_index(ID).reindex(ids[cand])
    Xc = X.loc[cand, cols].reset_index(drop=True)
    rcols = list(R.columns)
    for c in rcols:
        Xc[c] = R[c].to_numpy()
    yc, mc, Pc = tgt[cand], month[cand], P[cand]
    print(f'Rome candidates (off_sched > 1200 s): {cand.sum():,}; stamped share {yc.mean():.3f} (A/B/C/D {yc[~Pc].mean():.3f}, '
          f'P {yc[Pc].mean():.3f}) | regime coverage ' + ' '.join(f'{c} {Xc[c].notna().mean():.2f}' for c in rcols)
          + f' | {time.time() - t0:.0f}s, RSS {peak_rss_gb():.1f} GB', flush=True)
    del X
    cats = [c for c in W2_CATS if c in cols]
    res = {}
    for arm, use in (('base', cols), ('regime', cols + rcols)):
        oofs = []
        for seed in (0, 1):
            p = np.zeros(len(yc))
            for m in range(1, 13):
                tr, va = mc != m, mc == m
                mdl = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
                                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                                     seed=seed, verbose=-1, num_threads=4),
                                lgb.Dataset(Xc.loc[tr, use], yc[tr], categorical_feature=cats), num_boost_round=400)
                p[va] = mdl.predict(Xc.loc[va, use])
            oofs.append(p)
        p = np.mean(oofs, axis=0); res[arm] = p
        brier = lambda k: np.mean((p[k] - yc[k]) ** 2)
        print(f'{arm:7s} AUC {roc_auc_score(yc, p):.4f}  Brier {brier(np.ones(len(yc), bool)):.4f} | A/B/C/D: AUC '
              f'{roc_auc_score(yc[~Pc], p[~Pc]):.4f} Brier {brier(~Pc):.4f} | P: AUC {roc_auc_score(yc[Pc], p[Pc]):.4f} '
              f'Brier {brier(Pc):.4f} | per seed AUC ' + ' '.join(f'{roc_auc_score(yc, q):.4f}' for q in oofs), flush=True)
    np.savez(OUT / 't01_rome_stamp_oof.npz', cand=np.flatnonzero(cand), y=yc, month=mc, P=Pc, **res)
    print(f'done {time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
