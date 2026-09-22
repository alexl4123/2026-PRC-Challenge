"""One unit of the entity-embedding hyperparameter sweep.

TWO STAGES. Run `fixed` once per fold, then `sweep` once per (config, seed, fold).

    python cluster/nnsweep_task.py --campaign nnsweep --stage fixed  --group A --index 0
    python cluster/nnsweep_task.py --campaign nnsweep --stage sweep  --group A --index 0 --device cuda

WHY TWO STAGES

Everything except the network is identical across the 22 arms: the design matrix,
the fold's numeric re-expression, the tail classifier, the tail anchor, and the
LightGBM reference the residual correlation is measured against. Recomputing all
of that 88 times would spend most of the campaign rebuilding the same arrays --
and the LightGBM reference alone is a 1,400-round, 511-leaf fit. The `fixed`
stage computes it once per fold and caches it under

    cache/cv/<campaign>/_fixed/

so a `sweep` task starts in seconds and does nothing but train a network. That
is also what makes the comparison clean: every arm is spliced into literally the
same fixed components, so a measured difference is the network and nothing else.

WHERE THE OUTPUT GOES

    cache/cv/<campaign>/<cfg>__seed<NN>__m<MM>.npy   the full mixture prediction
    cache/cv/<campaign>/_bulk/<same name>.npy        the raw NN bulk prediction
    cache/cv/<campaign>/_meta/<same name>.json       metrics, timings, curve

The flat layout of the first one is deliberate: cluster/collect.py globs
`*.npy` NON-recursively in the campaign directory, so it scores the sweep arms
exactly as it scores every other campaign, and the `_`-prefixed subdirectories
are invisible to it. collect.py requires all twelve months, so it is the right
tool for the `confirm` group and reports the 4-fold screening groups as
INCOMPLETE -- use cluster/nnsweep_collect.py for those.

TWO REASONS A SWEEP ARM IS NOT DIRECTLY COMPARABLE TO THE 286.66 FROM THE `nn`
CAMPAIGN, and why the `base` arm exists:

  1. The tail classifier here is pinned at seed 0 for every arm and every NN
     seed. cluster/nn_task.py reseeds it with the NN seed, so its 3-seed spread
     contains classifier variance that the sweep deliberately excludes.
  2. The screening groups use four folds, not twelve, and the four are not the
     twelve's average.

So always read a sweep arm against the `base` arm of the SAME campaign. That is
what the paired bootstrap in nnsweep_collect.py does.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from cluster.nnsweep_configs import (CONFIGS, group, needed_missing_modes,
                                     prep_months, registry_fingerprint, unit)
from prc.config import CACHE, TAIL_CAP

# The tail classifier and the LightGBM reference are held at ONE seed for the
# whole campaign. Varying them would put variance into every arm that has
# nothing to do with the network.
FIXED_SEED = 0


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def campaign_dir(campaign, root=None):
    d = (Path(root) if root else CACHE / 'cv') / campaign
    d.mkdir(parents=True, exist_ok=True)
    return d


def fixed_dir(campaign, root=None):
    d = campaign_dir(campaign, root) / '_fixed'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_atomic(path, arr):
    """np.save through a temp file. Two tasks racing on one path must never be
    able to leave a half-written array that loads as garbage.

    The temp name has to KEEP the .npy suffix: np.save appends '.npy' to any
    filename that does not already end in it, so a temp called
    'codes__m12.npy.tmp1234' is actually written as
    'codes__m12.npy.tmp1234.npy' and the os.replace below then fails on a path
    that was never created. Putting the pid before the suffix avoids that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp{os.getpid()}{path.suffix}')
    np.save(tmp, arr)
    os.replace(tmp, path)


def _write_json_atomic(path, obj):
    """Same contract as _save_atomic, for the per-run metadata record.

    Creates the parent directory for the same reason: the first task to reach a
    campaign's _meta/ directory finds it absent, and Path.write_text does not
    create it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp{os.getpid()}{path.suffix}')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# design matrix
# ---------------------------------------------------------------------------

def load_design(source):
    """(X, y, month, cols, clf_cols).

    'assemble' is cluster/cv_task.assemble() -- exactly what produced the 286.66
    reference, and what every cluster task uses. It loads the whole 4.17M-row
    movement table and peaks around 8 GB, which is fine on a 2 TB node and is
    NOT fine on the laptop.

    'altfam' is prc.models.altfam.design(), the disk-cached build that exp40-53
    use. It follows assemble() step for step but loads a slim column subset for
    the congestion pass, peaking at ~1.6 GB from cache. Use it for local smoke
    tests. On the cluster it would need building first, and four prep tasks
    racing on one parquet write is a corruption risk, so it is NOT the default.
    """
    if source == 'assemble':
        from cluster.cv_task import CONFIGS as FEATCFG, assemble
        X, y, month, groups = assemble()
    elif source == 'altfam':
        from cluster.cv_task import CONFIGS as FEATCFG
        from prc.models.altfam import design
        X, y, month, groups = design()
    else:
        raise ValueError(source)
    cols = [c for g in FEATCFG['full'] for c in groups[g]]
    clf_cols = [c for g in FEATCFG['base'] for c in groups[g]]
    return X, np.asarray(y, float), np.asarray(month), cols, clf_cols


# ---------------------------------------------------------------------------
# stage: fixed
# ---------------------------------------------------------------------------

def stage_fixed(a):
    import lightgbm as lgb

    from cluster.study_task import clf_params, fit_bulk
    from prc.features import CATS
    from prc.models.altfam import numeric_design
    from prc.models.embed_nn import remap_unseen

    months = prep_months(a.group) if a.months is None else a.months
    if a.index is not None:
        if not 0 <= a.index < len(months):
            sys.exit(f'--index {a.index} out of range: group {a.group!r} has '
                     f'{len(months)} folds, so --array=0-{len(months) - 1}')
        months = [months[a.index]]
    modes = needed_missing_modes(a.group) if a.modes is None else a.modes

    fd = fixed_dir(a.campaign, a.out_root)
    todo = [m for m in months if not (fd / f'manifest__m{m:02d}.json').exists()]
    if not todo:
        print(f'all manifests present for months {months}, nothing to do')
        return
    print(f'prep months {todo}  modes {modes}  design={a.design}', flush=True)

    t0 = time.time()
    X, y, month, cols, clf_cols = load_design(a.design)
    cats = [c for c in CATS if c in cols]
    ccats = [c for c in CATS if c in clf_cols]
    print(f'design {X.shape[0]:,} x {X.shape[1]}  ({len(cols)} regressor cols, '
          f'{len(clf_cols)} classifier cols)  ({time.time() - t0:.0f}s)', flush=True)

    for m in todo:
        tm = time.time()
        va = month == m
        trn = ~va
        if a.subsample:
            # SMOKE-TEST ONLY. Recorded in the manifest so a subsampled prep can
            # never be mistaken for a real one downstream.
            rng = np.random.default_rng(0)
            keep = np.zeros(len(y), bool)
            keep[rng.choice(np.flatnonzero(trn), min(a.subsample, int(trn.sum())),
                            replace=False)] = True
            trn = keep
            kv = np.flatnonzero(va)
            va = np.zeros(len(y), bool)
            va[kv[:min(a.subsample // 4, len(kv))]] = True
        bulk_trn = trn & (y <= TAIL_CAP)
        print(f'  month {m:02d}: {int(trn.sum()):,} train ({int(bulk_trn.sum()):,} bulk) '
              f'-> {int(va.sum()):,} val', flush=True)

        # --- the numeric design, remapped for THIS fold --------------------
        # numeric_design returns SIX values (codes, cards, Z, cat_cols,
        # num_cols, stats); its docstring says four, which is wrong.
        codes = None
        for mode in modes:
            c, _cards, Z = numeric_design(X[cols], cols, fit_mask=bulk_trn,
                                          missing=mode)[:3]
            if codes is None:
                # Codes do not depend on the fill mode, only on the fold. The
                # unseen-level remap is applied ONCE here and cached, so a sweep
                # task can skip it; remap_unseen is idempotent either way.
                codes = remap_unseen(c, bulk_trn)
                _save_atomic(fd / f'codes__m{m:02d}.npy', codes.astype(np.int32))
            _save_atomic(fd / f'Z{mode}__m{m:02d}.npy', Z.astype(np.float32))
            del c, Z
        _save_atomic(fd / f'yall__m{m:02d}.npy', y.astype(np.float64))
        _save_atomic(fd / f'trn__m{m:02d}.npy', trn)
        _save_atomic(fd / f'va__m{m:02d}.npy', va)
        print(f'    numeric design cached ({time.time() - tm:.0f}s)', flush=True)

        # --- the fixed tail components -------------------------------------
        yt = y[trn]
        n_tail = int((yt > TAIL_CAP).sum())
        if n_tail == 0:
            sys.exit(f'month {m}: no tail rows in the training fold, so the tail '
                     f'mean is undefined. With --subsample this is expected; '
                     f'raise it or drop --subsample.')
        tail_mean = float(yt[yt > TAIL_CAP].mean())
        tc = time.time()
        mc = lgb.train(clf_params(FIXED_SEED, a.threads),
                       lgb.Dataset(X.loc[trn, clf_cols], (yt > TAIL_CAP).astype(int),
                                   categorical_feature=ccats), num_boost_round=300)
        ptail = mc.predict(X.loc[va, clf_cols])
        _save_atomic(fd / f'ptail__m{m:02d}.npy', np.asarray(ptail, np.float64))
        _save_atomic(fd / f'tailmean__m{m:02d}.npy', np.array([tail_mean]))
        _save_atomic(fd / f'offsched__m{m:02d}.npy',
                     X.loc[va, 'off_sched'].to_numpy(dtype=np.float64))
        print(f'    tail classifier: {n_tail} positives, mean p {ptail.mean():.2e}, '
              f'tail mean {tail_mean:,.0f}s  ({time.time() - tc:.0f}s)', flush=True)

        # --- the LightGBM reference the correlation is measured against -----
        # study_task.fit_bulk('mix', ...) IS the production bulk regressor
        # (511 leaves, 1,400 rounds, lr 0.025 -- exp50's winner). Imported
        # rather than copied so the two can never drift.
        lgb_secs = None
        if not a.skip_lgbref:
            tl = time.time()
            pb = fit_bulk('mix', X, y, trn, cols, cats, FIXED_SEED, a.threads, va=va)
            _save_atomic(fd / f'lgbbulk__m{m:02d}.npy', np.asarray(pb, np.float32))
            lgb_secs = time.time() - tl
            bulk_va = y[va] <= TAIL_CAP
            print(f'    lgb reference bulk RMSE '
                  f'{np.sqrt(np.mean((y[va][bulk_va] - pb[bulk_va]) ** 2)):.2f}  '
                  f'({lgb_secs:.0f}s)', flush=True)

        _write_json_atomic(fd / f'manifest__m{m:02d}.json', {
            'month': int(m), 'group': a.group, 'design': a.design,
            'fingerprint': registry_fingerprint(a.group),
            'n_rows': int(len(y)), 'n_train': int(trn.sum()),
            'n_bulk_train': int(bulk_trn.sum()), 'n_val': int(va.sum()),
            'n_cat': int(codes.shape[1]), 'n_num': int(len(cols) - codes.shape[1]),
            'modes': list(modes), 'tail_mean': tail_mean,
            'n_tail_train': n_tail, 'clf_seed': FIXED_SEED,
            'lgbref': not a.skip_lgbref, 'lgbref_secs': lgb_secs,
            'subsample': int(a.subsample), 'secs': time.time() - tm,
        })
        print(f'  month {m:02d} done ({time.time() - tm:.0f}s)', flush=True)
    print(f'prep complete ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------
# stage: sweep
# ---------------------------------------------------------------------------

def _require_fixed(fd, m, mode, group_name):
    man = fd / f'manifest__m{m:02d}.json'
    if not man.exists():
        sys.exit(
            f'MISSING PREREQUISITE: {man}\n'
            f'The fixed components for fold {m} have not been built. Run the prep\n'
            f'stage first:\n'
            f'  sbatch --array=0-{len(prep_months(group_name)) - 1} '
            f'cluster/nnsweep_array.sbatch <campaign> fixed {group_name}')
    meta = json.loads(man.read_text())
    if mode not in meta['modes']:
        sys.exit(f'fold {m} was prepped with modes {meta["modes"]} but this config '
                 f"needs '{mode}'. Delete {fd}/manifest__m{m:02d}.json and re-run "
                 f'the prep stage.')
    for f in [f'codes__m{m:02d}.npy', f'Z{mode}__m{m:02d}.npy',
              f'yall__m{m:02d}.npy', f'trn__m{m:02d}.npy', f'va__m{m:02d}.npy',
              f'ptail__m{m:02d}.npy', f'tailmean__m{m:02d}.npy',
              f'offsched__m{m:02d}.npy']:
        if not (fd / f).exists():
            sys.exit(f'fold {m} manifest exists but {f} does not -- the prep stage '
                     f'was interrupted. Delete {fd}/manifest__m{m:02d}.json and '
                     f're-run it.')
    return meta


def stage_sweep(a):
    from prc.models.altfam import mixture
    from prc.models.embed_nn import resolve_device

    cfg_name, seed, m = unit(a.group, a.index)
    cfg = CONFIGS[cfg_name]

    # Resolve the device FIRST -- before the stale-arm check, before the
    # idempotent skip, before anything. A whole array submitted to -p gpu
    # without --gres=gpu:N sees no devices, and if the skip path runs first
    # every already-finished unit exits 0 and the misconfiguration only
    # surfaces on the first unfinished one, hours later. Failing here means the
    # very first task says so, whatever state the campaign is in.
    dev = resolve_device(None if a.device == 'auto' else a.device)

    d = campaign_dir(a.campaign, a.out_root)
    (d / '_bulk').mkdir(exist_ok=True)
    (d / '_meta').mkdir(exist_ok=True)
    stem = f'{cfg_name}__seed{seed:02d}__m{m:02d}'
    out, out_bulk = d / f'{stem}.npy', d / '_bulk' / f'{stem}.npy'
    out_meta = d / '_meta' / f'{stem}.json'
    cur_params = {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in cfg.items() if k != 'note'}
    if out_meta.exists():
        # A config edited halfway through a campaign would put two different
        # models under one arm name, and nothing downstream could tell. Every
        # unit records the exact parameters it ran; refuse if they moved.
        prev = json.loads(out_meta.read_text()).get('params')
        if prev is not None and prev != cur_params:
            sys.exit(f'STALE ARM {stem}: this file was written for\n  {prev}\n'
                     f'but {cfg_name!r} now means\n  {cur_params}\n'
                     f'Revert the edit to cluster/nnsweep_configs.py, or run under a '
                     f'NEW --campaign name.')
        if out.exists() and out_bulk.exists():
            print(f'{stem} complete, skipping')
            return

    print(f'{stem}  device={dev}  cfg={ {k: v for k, v in cfg.items() if k != "note"} }',
          flush=True)

    fd = fixed_dir(a.campaign, a.out_root)
    meta = _require_fixed(fd, m, cfg['missing'], a.group)

    t0 = time.time()
    codes = np.load(fd / f'codes__m{m:02d}.npy')
    Z = np.load(fd / f'Z{cfg["missing"]}__m{m:02d}.npy')
    y = np.load(fd / f'yall__m{m:02d}.npy')
    trn = np.load(fd / f'trn__m{m:02d}.npy')
    va = np.load(fd / f'va__m{m:02d}.npy')
    ptail = np.load(fd / f'ptail__m{m:02d}.npy')
    offsched = np.load(fd / f'offsched__m{m:02d}.npy')
    tail_mean = float(np.load(fd / f'tailmean__m{m:02d}.npy')[0])
    if not (len(y) == len(codes) == len(Z) == len(trn) == len(va)):
        sys.exit(f'fold {m}: cached arrays disagree on length '
                 f'({len(y)}, {len(codes)}, {len(Z)}, {len(trn)}, {len(va)})')
    if not (int(va.sum()) == len(ptail) == len(offsched)):
        sys.exit(f'fold {m}: {int(va.sum())} validation rows but {len(ptail)} tail '
                 f'probabilities and {len(offsched)} off_sched values')
    bulk_trn = trn & (y <= TAIL_CAP)
    print(f'  loaded fixed components ({time.time() - t0:.0f}s): '
          f'{int(bulk_trn.sum()):,} bulk train -> {int(va.sum()):,} val', flush=True)

    from cluster.nnsweep_model import fit_sweep
    yva = y[va]
    bulk_va = yva <= TAIL_CAP
    lgb_path = fd / f'lgbbulk__m{m:02d}.npy'
    p_lgb = np.load(lgb_path).astype(np.float64) if lgb_path.exists() else None

    curve = []

    def hook(ep, train_loss, pred):
        r = float(np.sqrt(np.mean((yva[bulk_va] - pred[bulk_va]) ** 2)))
        c = (float(np.corrcoef(yva[bulk_va] - pred[bulk_va],
                               yva[bulk_va] - p_lgb[bulk_va])[0, 1])
             if p_lgb is not None else None)
        curve.append({'epoch': ep, 'train_loss': train_loss, 'val_bulk_rmse': r,
                      'corr_lgb': c, 't': time.time() - t0})
        print(f'    epoch {ep:3d}/{cfg["epochs"]}  train loss(std) {train_loss:.5f}  '
              f'val bulk RMSE {r:8.2f}' + (f'  corr {c:.4f}' if c is not None else ''),
              flush=True)

    tfit = time.time()
    kw = {k: v for k, v in cfg.items() if k not in ('note', 'missing')}
    kw['hidden'] = tuple(kw['hidden'])
    pred_bulk, info = fit_sweep(codes, Z, y, bulk_trn, va, seed=seed,
                                threads=a.threads, device=str(dev),
                                eval_hook=hook, already_remapped=True,
                                log=lambda s: print(s, flush=True), **kw)
    fit_secs = time.time() - tfit

    pred = mixture(ptail, np.clip(pred_bulk, 0, None), offsched, tail_mean)
    _save_atomic(out_bulk, np.asarray(pred_bulk, np.float32))
    _save_atomic(out, np.asarray(pred, np.float32))

    res_nn = yva[bulk_va] - pred_bulk[bulk_va]
    rec = {
        'config': cfg_name, 'seed': int(seed), 'month': int(m),
        'group': a.group, 'params': cur_params,
        'note': cfg.get('note', ''),
        'n_val': int(va.sum()), 'n_val_bulk': int(bulk_va.sum()),
        'n_bulk_train': int(bulk_trn.sum()),
        # the mixture, i.e. what the model would actually score
        'rmse': float(np.sqrt(np.mean((yva - pred) ** 2))),
        'rmse_bulk': float(np.sqrt(np.mean((yva[bulk_va] - pred[bulk_va]) ** 2))),
        # the network alone, on bulk rows -- the quantity the sweep varies
        'nn_bulk_rmse': float(np.sqrt(np.mean(res_nn ** 2))),
        'corr_lgb': None, 'lgb_bulk_rmse': None, 'blend50_bulk_rmse': None,
        'n_params': info['n_params'], 'n_emb_params': info['n_emb_params'],
        'dims': info['dims'], 'cards': info['cards'], 'steps': info['steps'],
        'curve': curve, 'fit_secs': fit_secs, 'total_secs': time.time() - t0,
        'device': str(dev), 'subsample': meta.get('subsample', 0),
        'design': meta.get('design'), 'fingerprint': meta.get('fingerprint'),
    }
    if p_lgb is not None:
        res_lgb = yva[bulk_va] - p_lgb[bulk_va]
        blend = 0.5 * (pred_bulk[bulk_va] + p_lgb[bulk_va])
        rec['corr_lgb'] = float(np.corrcoef(res_nn, res_lgb)[0, 1])
        rec['lgb_bulk_rmse'] = float(np.sqrt(np.mean(res_lgb ** 2)))
        rec['blend50_bulk_rmse'] = float(np.sqrt(np.mean((yva[bulk_va] - blend) ** 2)))
    _write_json_atomic(out_meta, rec)

    print(f'{stem}: mixture RMSE {rec["rmse"]:.2f} | bulk {rec["rmse_bulk"]:.2f} | '
          f'NN-alone bulk {rec["nn_bulk_rmse"]:.2f}' +
          (f' | corr vs lgb {rec["corr_lgb"]:.4f} | 50/50 blend '
           f'{rec["blend50_bulk_rmse"]:.2f}' if p_lgb is not None else
           ' | NO lgb reference cached, correlation not measured') +
          f'  (fit {fit_secs:.0f}s, total {time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--stage', required=True, choices=['fixed', 'sweep'])
    ap.add_argument('--group', default='A')
    ap.add_argument('--index', type=int, default=None,
                    help='SLURM_ARRAY_TASK_ID; resolved by nnsweep_configs.unit()')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    ap.add_argument('--design', default='assemble', choices=['assemble', 'altfam'],
                    help='prep only; altfam is the laptop-sized cached build')
    ap.add_argument('--months', type=int, nargs='+', default=None,
                    help='prep only; overrides the group folds (local testing)')
    ap.add_argument('--modes', nargs='+', default=None, choices=['median', 'sentinel'],
                    help='prep only; overrides which numeric designs to cache')
    ap.add_argument('--skip-lgbref', action='store_true',
                    help='prep only; skip the 1,400-round LightGBM reference. '
                         'Residual correlation is then NOT measurable.')
    ap.add_argument('--subsample', type=int, default=0,
                    help='prep only; SMOKE TESTS. Keep this many training rows.')
    ap.add_argument('--out-root', default=None,
                    help='write the campaign here instead of cache/cv. The prep\n                         stage caches ~1.1 GB per fold, and this repository sits\n                         inside a Dropbox-synced tree, so point local smoke runs\n                         at a scratch directory rather than pushing gigabytes\n                         through the sync daemon (which was observed removing and\n                         restoring these files mid-run). Unused on the cluster.')
    a = ap.parse_args()

    if a.stage == 'sweep':
        if a.index is None:
            sys.exit('--index is required for --stage sweep')
        group(a.group)          # validates the group before anything expensive
        stage_sweep(a)
    else:
        stage_fixed(a)


if __name__ == '__main__':
    main()
