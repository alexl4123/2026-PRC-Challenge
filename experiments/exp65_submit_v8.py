"""EXP65 -- v8 = parent submission + the LightGBM / network 50/50 blend in the bulk branch.

WHAT CHANGES
------------
Exactly one thing relative to the parent (v7, or v6 if v7 loses on the board):
inside each of the five LightGBM seeds' mixtures, the bulk prediction becomes

    pb_mix = (1 - w) * pb_lgb_seed + w * pb_net,   w = 0.5 (fixed in advance)

where pb_net is the 3-seed average of cluster/nn_final.py. The global tail
classifier, the specialist inside P, the tail anchor `et` and the clip are all
the parent's. So the official delta measures the blend and nothing else.

THE EVIDENCE (12 leave-one-month-out folds, study campaign, v5/v6 LGB config)
    LightGBM alone                  286.94   bulk 234.80
    50/50 LGB + 3-seed network      283.06   bulk 230.52
    paired -3.94 +- 0.41  CI [-4.72, -3.09] | outside P -5.02 +- 0.26
    scored months 1+7  -4.78 +- 0.83
Expect about -3s live: that LGB was ONE seed, v6/v7 average five, which already
removes part of the variance the blend removes.

TWO STAGES, so later variants cost seconds, not half an hour
    --stage components   (~30 min, 4 threads, ~5.5 GB) fits exp54's five LightGBM
                         seeds and caches their per-seed ranking predictions at
                         FULL float64 precision in cache/final/lgb_components.npz,
                         together with v6's specialist probabilities.
    --stage assemble     (seconds) combines the cached pieces and builds the file.

THE REGRESSION CHECK -- run it before building anything
    --stage assemble --specialist v6 --w 0 --check-against 6
must reproduce outputs/understanding-accordion_v6.parquet EXACTLY (every row).
That proves the cached components are v6's own. With --specialist x it must
reproduce v7 instead (needs cache/final/spec_x_ranking.npz from exp64).
--check-against never writes a submission; it only compares.

    python experiments/exp65_submit_v8.py --stage components
    python experiments/exp65_submit_v8.py --stage assemble --specialist x  --w 0   --check-against 7
    python experiments/exp65_submit_v8.py --stage assemble --specialist x  --w 0.5 --version 8
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from prc.config import TARGET, TAIL_CAP, ID, CACHE, OUTPUTS
from prc.submit import TEAM
from prc.models.tail_specialist import apply_to_mixture

FINAL = CACHE / 'final'
COMP = FINAL / 'lgb_components.npz'
SPEC_X = FINAL / 'spec_x_ranking.npz'
NET = CACHE / 'nn_final'
N_SEEDS = 5


def stage_components(threads):
    import lightgbm as lgb
    from prc.features import build as build_base, align_categories, CATS
    from prc.congestion import build as build_cong
    from prc.weather import download, prepare, attach as attach_wx
    from prc.disruption import attach as attach_atfm
    from prc.io import load, peak_rss_gb
    from prc.models.tail_specialist import TailSpecialist

    def assemble(which, extra_cols):
        """Identical to experiments/exp54_submit_v6.py."""
        full = load(which)
        cong = build_cong(full)
        d = full[full.PHASE_mvt.eq('DEP')].reset_index(drop=True)
        del full
        X = build_base(d)
        X['aobt_missing'] = d.AOBT_3_flt.isna().astype('int8')
        X['lobt_missing'] = d.LOBT_flt.isna().astype('int8')
        X['off_lobt'] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds()
        X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
        X['log_off_aobt3'] = np.sign(X.off_aobt3) * np.log1p(X.off_aobt3.abs())
        base_cols = list(X.columns)
        wx = attach_wx(d, prepare(download()))
        atfm = attach_atfm(d)
        X = pd.concat([X, cong.set_axis(X.index), wx.set_axis(X.index),
                       atfm.set_axis(X.index)], axis=1)
        return d[extra_cols], X, base_cols

    t0 = time.time()
    tr_meta, Xtr, BASE = assemble('training', [TARGET])
    rk_meta, Xrk, _ = assemble('ranking', [ID])
    Xtr, Xrk = align_categories(Xtr, Xrk)
    FULL = list(Xtr.columns)
    y = tr_meta[TARGET].astype(float).to_numpy()
    os_rk = Xrk.off_sched.to_numpy()
    print(f'train {Xtr.shape}  ranking {Xrk.shape} | {time.time()-t0:.0f}s | '
          f'RSS {peak_rss_gb():.2f} GB', flush=True)

    # exp54's parameters, verbatim.
    REG = dict(objective='regression', metric='rmse', learning_rate=0.025, num_leaves=511,
               min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=threads, verbose=-1,
               deterministic=True, force_row_wise=True)
    CLF = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
               min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=threads, verbose=-1,
               deterministic=True, force_row_wise=True)
    SCALE = 12 / 11
    bulk = y <= TAIL_CAP
    is_tail = (y > TAIL_CAP).astype(int)
    tail_mean = y[y > TAIL_CAP].mean()
    rcat = [c for c in CATS if c in FULL]
    ccat = [c for c in CATS if c in BASE]

    apt_tr = Xtr['APT'].astype(str).to_numpy(); apt_rk = Xrk['APT'].astype(str).to_numpy()
    os_tr = Xtr['off_sched'].to_numpy()
    miss_tr = Xtr['aobt_missing'].to_numpy().astype(bool)
    miss_rk = Xrk['aobt_missing'].to_numpy().astype(bool)
    spec = TailSpecialist(threads=4).fit(Xtr, y, apt_tr, os_tr, miss_tr)
    p_spec_v6, in_P = spec.predict(Xrk, apt_rk, os_rk, miss_rk)
    print(f'v6 specialist: {int(in_P.sum()):,} ranking rows inside P', flush=True)

    pb = np.empty((N_SEEDS, len(Xrk))); pt = np.empty((N_SEEDS, len(Xrk)))
    for seed in range(N_SEEDS):
        t = time.time()
        r = dict(REG, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
        c = dict(CLF, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
        mb = lgb.train(r, lgb.Dataset(Xtr.loc[bulk, FULL], y[bulk], categorical_feature=rcat),
                       num_boost_round=int(1400 * SCALE))
        mc = lgb.train(c, lgb.Dataset(Xtr[BASE], is_tail, categorical_feature=ccat),
                       num_boost_round=int(300 * SCALE))
        pb[seed] = np.clip(mb.predict(Xrk[FULL]), 0, None)
        pt[seed] = mc.predict(Xrk[BASE])
        print(f'  seed {seed}: bulk mean {pb[seed].mean():.1f}s  ({time.time()-t:.0f}s)', flush=True)

    et = np.where(np.isfinite(os_rk) & (os_rk > TAIL_CAP), os_rk, tail_mean)
    FINAL.mkdir(parents=True, exist_ok=True)
    tmp = FINAL / '.lgb_components.tmp.npz'
    np.savez(tmp, ids=rk_meta[ID].to_numpy(), pb=pb, pt=pt, et=et, in_P=in_P,
             p_spec_v6=p_spec_v6, threads=threads)
    os.replace(tmp, COMP)
    print(f'saved {COMP} | peak RSS {peak_rss_gb():.2f} GB | '
          f'{(time.time()-t0)/60:.1f} min', flush=True)


def to_int32(pred):
    """Exactly what prc.submit.build does to a prediction vector (clip, then rint)."""
    return np.rint(np.clip(np.asarray(pred, float), 0.0, 200000.0)).astype(np.int32)


def stage_assemble(specialist, w, version, check_against, w_abc=None, w_d=None):
    if not COMP.exists():
        sys.exit(f'FATAL: {COMP} missing -- run --stage components first')
    c = np.load(COMP, allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)

    if specialist == 'v6':
        p_spec = c['p_spec_v6']
    else:
        if not SPEC_X.exists():
            sys.exit(f'FATAL: {SPEC_X} missing -- it is written by the v7 build '
                     '(experiments/exp64_submit_v7.py)')
        s = np.load(SPEC_X, allow_pickle=True)
        if not np.array_equal(s['ids'], ids) or not np.array_equal(s['in_P'].astype(bool), in_P):
            sys.exit('FATAL: spec_x_ranking.npz is not aligned with the components '
                     '(ids or P mask differ)')
        p_spec = s['p_spec']

    # Per-row network weight. Route D = no NM off-block record (AOBT_3_flt missing).
    # v9: the network deserves ~0.65 on ordinary flights (stable across 1/2/3 LightGBM
    # seeds) but route D's optimum is unstable, so it keeps v8's value. See the log.
    if w_abc is not None or w_d is not None:
        from prc.io import load
        rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt'])
        if not np.array_equal(rk[ID].to_numpy(), ids):
            sys.exit('FATAL: ranking order differs from the cached components')
        route_d = rk['AOBT_3_flt'].isna().to_numpy()
        wv = np.where(route_d, w if w_d is None else w_d, w if w_abc is None else w_abc)
        print(f'per-route network weight: {w if w_abc is None else w_abc} on {int((~route_d).sum()):,} '
              f'rows with an NM record, {w if w_d is None else w_d} on {int(route_d.sum()):,} without')
    else:
        wv = np.full(len(ids), w)
    w = float(wv.max())                               # "> 0" test below: is the network used at all?

    if w > 0:
        seeds = sorted(NET.glob('nn_final_seed*.npy'))
        if len(seeds) != 3:
            sys.exit(f'FATAL: expected 3 network seeds in {NET}, found {len(seeds)} -- '
                     'run cluster/nn_final.sbatch and rsync cache/nn_final/ back')
        if not np.array_equal(np.load(NET / 'ranking_ids.npy', allow_pickle=True), ids):
            sys.exit('FATAL: network predictions are in a different ranking order')
        net = np.mean([np.load(f).astype(float) for f in seeds], axis=0)
        print(f'network: {len(seeds)} seeds, mean {net.mean():.1f}s, corr with LGB bulk '
              f'{np.corrcoef(net, pb.mean(0))[0,1]:.4f}')
    else:
        net = np.zeros(len(ids))

    preds = []
    for seed in range(pb.shape[0]):
        p = apply_to_mixture(pt[seed], p_spec, in_P)
        pbm = (1 - wv) * pb[seed] + wv * net
        preds.append(np.clip(p * et + (1 - p) * pbm, 0, None))
    pred = np.mean(preds, axis=0)

    if check_against is not None:
        parent = OUTPUTS / f'{TEAM}_v{check_against}.parquet'
        par = pd.read_parquet(parent).set_index('MVT_ID_mvt')['TAXITIME_SEC_mvt']
        mine = pd.Series(to_int32(pred), index=pd.Index(ids, name='MVT_ID_mvt'))
        par = par.reindex(mine.index)
        diff = (mine.to_numpy() != par.to_numpy())
        print(f'CHECK vs {parent.name} (specialist={specialist}, w={w}): '
              f'{int(diff.sum()):,} of {len(mine):,} rows differ'
              + (f' | inside P {int(diff[in_P].sum())}, outside P {int(diff[~in_P].sum())}, '
                 f'max |diff| {int(np.abs(mine.to_numpy()-par.to_numpy()).max())}s'
                 if diff.any() else ' -- IDENTICAL'))
        return

    if w > 0:
        base = np.mean([np.clip(apply_to_mixture(pt[s], p_spec, in_P) * et
                                + (1 - apply_to_mixture(pt[s], p_spec, in_P)) * pb[s], 0, None)
                        for s in range(pb.shape[0])], axis=0)
        d = pred - base
        print(f'blend vs parent: rows moved >1s {int((np.abs(d) > 1).sum()):,} | '
              f'mean shift {d.mean():+.2f}s | |shift| median {np.median(np.abs(d)):.1f}s, '
              f'p99 {np.percentile(np.abs(d), 99):.0f}s')
    from prc.submit import build as build_submission
    path = build_submission(ids, pred, version=version, clip=(0.0, 200000.0))
    print(f'built {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['components', 'assemble'])
    ap.add_argument('--specialist', default='x', choices=['v6', 'x'])
    ap.add_argument('--w', type=float, default=0.5, help='weight on the network (0.5 fixed in advance)')
    ap.add_argument('--w-abc', type=float, default=None,
                    help='network weight on rows WITH an NM off-block record (routes A/B/C); default --w')
    ap.add_argument('--w-d', type=float, default=None,
                    help='network weight on rows WITHOUT one (route D); default --w')
    ap.add_argument('--version', type=int, default=8)
    ap.add_argument('--check-against', type=int, default=None,
                    help='compare with outputs/<team>_vN.parquet instead of building')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.stage == 'components':
        if COMP.exists():
            print(f'{COMP} exists, skipping (delete it to refit)')
            return
        stage_components(a.threads)
    else:
        stage_assemble(a.specialist, a.w, a.version, a.check_against, a.w_abc, a.w_d)


if __name__ == '__main__':
    main()
