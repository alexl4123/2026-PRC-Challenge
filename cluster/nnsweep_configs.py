"""The hyperparameter sweep's configuration registry and its index mapping.

WHY A REGISTRY RATHER THAN A GRID

A competitor's 160-trial Optuna run produced their WORST submission, and our own
LightGBM tuning (exp50) bought ~2 s from an eight-point search. The evidence says
the payoff here is in knowing which DIRECTION each axis wants, not in locating a
joint optimum. So stage A is a one-factor-at-a-time ladder around the shipped
configuration plus three deliberate interaction corners, and stage B is a small
refinement written only after stage A has been read.

Every config below is a full dict, not a delta, so a filename in the campaign
directory is enough to reconstruct exactly what was run. `BASE` is the shipped
configuration and is included as an arm: it is the control that every other arm
is paired against, inside the same campaign, on the same folds, with the same
fixed tail components. Do NOT compare a sweep arm against the 286.66 from the
`nn` campaign -- see cluster/nnsweep_task.py for the two reasons that number is
not directly comparable.

NAMING. A config name must not contain '__'. cluster/collect.py splits filenames
on '__' into <arm>__seed<NN>__m<MM>, so a double underscore in a name silently
reassigns the arm/seed/month fields.

THE INDEX MAPPING lives here rather than in the sbatch file. cluster/nn_array.sbatch
does `MONTH=$(( i % 12 + 1 ))` in bash, which means the mapping exists twice --
once in bash and once in whatever reads the results -- and the two can drift. Here
the task resolves its own index, `python cluster/nnsweep_configs.py` prints the
plan, and `--verify` proves the mapping is a bijection onto the full product.
"""
import itertools
import json

# Screening folds. NOT months 1, 7 or 10.
#
#   1 and 7 are the two scored months and are quoted in every report we have
#     written; choosing hyperparameters on them is selection on the evaluation
#     set and would make the reported number optimistic by an unknown amount.
#   10 is the third month of the exp45/exp46 screening set, so the alternative
#     model families (CatBoost, XGBoost, the NN itself, the blend weights) have
#     already been compared on it. Reusing it compounds the same selection.
#   4 was used by exp50 to size the LightGBM regressor. Avoided for the same
#     reason even though the model family differs.
#
# 3, 6, 9, 12 is one month per quarter: it keeps all four seasons in the
# screening set, so a configuration that only suits summer congestion cannot win
# on seasonal luck. Bulk RMSE of the shipped model on these four (202.2, 245.6,
# 240.6, 218.4; mean 226.7) brackets the 12-month mean of 232.9 with a hard
# month (6) and an easy one (3), and together they hold 697,218 validation rows
# -- a third of the data, which is ample for RANKING.
SCREEN_MONTHS = (3, 6, 9, 12)
ALL_MONTHS = tuple(range(1, 13))

# The shipped configuration, as cluster/nn_task.py runs it today.
BASE = dict(
    epochs=12,
    batch=8192,
    lr=3e-3,
    weight_decay=1e-5,
    hidden=(512, 256, 128),
    dropout=0.10,
    emb_cap=32,          # min(cap, max(floor, round(mult * card**pow)))
    emb_mult=1.6,
    emb_pow=0.56,
    emb_floor=3,
    schedule='onecycle',
    pct_start=0.25,
    loss='mse',
    huber_delta=1.0,
    missing='median',
)


def cfg(**kw):
    """A full config: BASE with the named fields replaced. Unknown key -> error."""
    bad = set(kw) - set(BASE) - {'note'}
    if bad:
        raise KeyError(f'unknown config field(s): {sorted(bad)}')
    out = dict(BASE)
    out.update(kw)
    return out


# ---------------------------------------------------------------------------
# STAGE A -- 22 arms, one axis at a time plus three interaction corners.
# ---------------------------------------------------------------------------
CONFIGS = {
    'base': cfg(note='the shipped configuration -- the control'),

    # --- training length -----------------------------------------------------
    # The single most likely win: the training loss went 0.339 -> 0.235 over 12
    # epochs and was still descending. CAVEAT, and it matters for how this is
    # read: with OneCycleLR the learning rate at epoch 12 is max_lr/25/1e4 =
    # 1.2e-8, i.e. the run has already fully annealed. A 12-epoch OneCycle run
    # is a COMPLETE training run, not a truncated one, so "still descending"
    # does not license extrapolating to 40 epochs -- it has to be re-run with
    # the schedule stretched over the longer budget, which is what these do.
    'ep24': cfg(epochs=24, note='2x training length'),
    'ep40': cfg(epochs=40, note='3.3x training length'),

    # --- learning rate -------------------------------------------------------
    'lr1': cfg(lr=1e-3, note='max_lr 1e-3 (lower bracket)'),
    'lr6': cfg(lr=6e-3, note='max_lr 6e-3 (upper bracket)'),
    # Longer runs usually want a different peak rate: more steps at high LR is
    # the mechanism by which extra epochs help, so epochs x lr is the one
    # interaction worth spending arms on.
    'ep40lr6': cfg(epochs=40, lr=6e-3, note='epochs x lr corner'),

    # --- schedule ------------------------------------------------------------
    'cosine': cfg(schedule='cosine', pct_start=0.05,
                  note='5% linear warmup then cosine to 0, no OneCycle div/final_div'),
    'pct10': cfg(pct_start=0.10, note='OneCycle, warmup 10% instead of 25%'),

    # --- width and depth -----------------------------------------------------
    'narrow': cfg(hidden=(256, 128), note='narrower, ~4x fewer MLP params'),
    'wide': cfg(hidden=(1024, 512, 256), note='wider'),
    'deep': cfg(hidden=(512, 512, 256, 128), note='one more layer at the top'),

    # --- dropout and weight decay -------------------------------------------
    # weight_decay=1e-5 in AdamW is decoupled and is effectively no
    # regularisation at all; 1e-2 is the value AdamW was designed around.
    'drop0': cfg(dropout=0.0, note='no dropout'),
    'drop20': cfg(dropout=0.20, note='double dropout'),
    'wd1e4': cfg(weight_decay=1e-4, note='10x weight decay'),
    'wd1e2': cfg(weight_decay=1e-2, note='1000x weight decay (the AdamW default scale)'),

    # --- batch size ----------------------------------------------------------
    # Batch is CONFOUNDED WITH TRAINING LENGTH at fixed epochs: steps =
    # epochs * n / batch, so b2k takes 4x the optimiser steps of base and b32k a
    # quarter. The learning rate is scaled by sqrt(batch ratio) to hold the
    # gradient-noise scale roughly fixed. If b2k wins, the follow-up question is
    # whether it was the batch or the step count -- ep40 answers that.
    'b2k': cfg(batch=2048, lr=1.5e-3, note='batch 2048, lr/2 (sqrt scaling)'),
    'b32k': cfg(batch=32768, lr=6e-3, note='batch 32768, lr*2 (sqrt scaling)'),

    # --- the embedding-width rule -------------------------------------------
    # The cap is a guess. It binds on exactly two tables: STAND_mvt (1,899
    # levels -> the rule asks for 110) and AIRCRAFT_OPERATOR_flt (676 -> 61).
    # At cap 128 nothing is capped, so this ladder spans "much tighter than
    # shipped" to "the rule left alone".
    'cap16': cfg(emb_cap=16, note='embedding cap 16'),
    'cap64': cfg(emb_cap=64, note='embedding cap 64'),
    'cap128': cfg(emb_cap=128, note='embedding cap 128 = effectively uncapped'),

    # --- two things that are not hyperparameters in the usual sense ----------
    # exp47 measured sentinel fill at -1.17 +- 0.47 solo, but it also raised the
    # residual correlation with LightGBM from 0.9246 to 0.9300 -- the network
    # became more tree-like exactly where it was most different. That trade is
    # the whole question this sweep exists to answer, so it gets an arm at full
    # scale. The shipped network uses 'median'.
    'sent': cfg(missing='sentinel', note="NaN -> -10 sentinel instead of the median"),
    # The most likely source of DECORRELATION on this list. Huber on the
    # standardised target (delta=1.0 is ~1 sd ~ 450 s) fits something closer to
    # a conditional median, so it should score worse on RMSE and be wrong in a
    # different place. At rho 0.956 a second member is worth ~1 s; at rho 0.90
    # it is worth ~7 s. A config that loses 3 s solo and drops rho to 0.90 is
    # the better one.
    'huber': cfg(loss='huber', note='Huber loss on the standardised target'),
}

STAGE_A = ('base', 'ep24', 'ep40', 'lr1', 'lr6', 'ep40lr6', 'cosine', 'pct10',
           'narrow', 'wide', 'deep', 'drop0', 'drop20', 'wd1e4', 'wd1e2',
           'b2k', 'b32k', 'cap16', 'cap64', 'cap128', 'sent', 'huber')

# ---------------------------------------------------------------------------
# Groups. A group is what one sbatch submission covers: which configs, over
# which folds, at which seeds. `confirm` is a PLACEHOLDER -- edit its `configs`
# after reading the stage-A collector output, then re-run
# `python cluster/nnsweep_configs.py` to get the new --array range.
# ---------------------------------------------------------------------------
GROUPS = {
    'A': dict(configs=STAGE_A, months=SCREEN_MONTHS, seeds=(0,),
              note='coarse screen, 22 arms x 4 folds x 1 seed'),
    # Stage B is written after stage A. Keeping it here with the base arm only
    # means a premature submission costs 4 tasks, not 88.
    'B': dict(configs=('base',), months=SCREEN_MONTHS, seeds=(0,),
              note='REFINEMENT -- edit configs after reading stage A'),
    # THE NOISE FLOOR, and the reason stage A can be read at all. The paired
    # bootstrap in nnsweep_collect.py resamples ROWS, so it measures row-sampling
    # noise and says nothing about SEED noise -- yet stage A compares 22 arms at
    # one seed each. These eight extra tasks (seed 0 is already in group A, so
    # only seeds 1 and 2 are new) give the spread of the SAME configuration
    # across seeds on exactly the screening protocol. Any stage-A difference
    # smaller than that spread is not a difference. Run it alongside stage A.
    'seed': dict(configs=('base',), months=SCREEN_MONTHS, seeds=(0, 1, 2),
                 note='seed-noise floor for the screening protocol'),
    # The decision run. Two arms over all twelve folds at three seeds is what
    # actually gets believed; the screen only nominates candidates.
    'confirm': dict(configs=('base',), months=ALL_MONTHS, seeds=(0, 1, 2),
                    note='12-fold x 3-seed confirmation -- edit configs first'),
    # Local CPU smoke test: every stage-A arm, one fold, run with --subsample.
    # NOT for the cluster. cluster/nnsweep_smoke.py drives it.
    'smoke': dict(configs=STAGE_A, months=(12,), seeds=(0,),
                  note='LOCAL smoke test -- every arm, one subsampled fold'),
}


def group(name):
    if name not in GROUPS:
        raise KeyError(f'unknown group {name!r}; have {sorted(GROUPS)}')
    g = GROUPS[name]
    missing = [c for c in g['configs'] if c not in CONFIGS]
    if missing:
        raise KeyError(f'group {name!r} names configs that do not exist: {missing}')
    return g


def units(name):
    """Every (config, seed, month) the group covers, in index order.

    Month varies FASTEST, then seed, then config. That ordering means a config's
    folds finish together, so a campaign stopped halfway still yields complete,
    usable arms rather than every arm missing a fold.
    """
    g = group(name)
    return [(c, s, m) for c in g['configs'] for s in g['seeds'] for m in g['months']]


def n_units(name):
    return len(units(name))


def unit(name, i):
    """Resolve one array index. Raises rather than wrapping around."""
    u = units(name)
    if not 0 <= i < len(u):
        raise IndexError(f'index {i} out of range for group {name!r}: '
                         f'{len(u)} units, so --array=0-{len(u) - 1}')
    return u[i]


def prep_months(name):
    return list(group(name)['months'])


def needed_missing_modes(name):
    """Which numeric designs the prep stage has to cache for this group.

    Each mode is ~534 MB per month on disk, so only build what is used.
    """
    return sorted({CONFIGS[c]['missing'] for c in group(name)['configs']})


def registry_fingerprint(name):
    """Stable hash of the exact configs a group will run.

    Written into the campaign manifest by the prep stage and re-checked by every
    sweep task, so editing a config after half a campaign has run is caught
    instead of producing a directory in which two arms with the same name mean
    two different models.
    """
    import hashlib
    g = group(name)
    payload = json.dumps(
        {'months': list(g['months']), 'seeds': list(g['seeds']),
         'configs': {c: {k: list(v) if isinstance(v, tuple) else v
                         for k, v in CONFIGS[c].items() if k != 'note'}
                     for c in g['configs']}},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def verify():
    """The index mapping must be a bijection onto the full product. Loud if not."""
    ok = True
    for name in GROUPS:
        g = group(name)
        want = set(itertools.product(g['configs'], g['seeds'], g['months']))
        got = [unit(name, i) for i in range(n_units(name))]
        problems = []
        if len(got) != len(want):
            problems.append(f'{len(got)} indices for {len(want)} combinations')
        if len(set(got)) != len(got):
            dupes = {t for t in got if got.count(t) > 1}
            problems.append(f'{len(got) - len(set(got))} duplicate(s), e.g. {sorted(dupes)[:3]}')
        if set(got) != want:
            problems.append(f'missing {sorted(want - set(got))[:3]}, '
                            f'extra {sorted(set(got) - want)[:3]}')
        for i in (-1, n_units(name)):
            try:
                unit(name, i)
                problems.append(f'index {i} did not raise')
            except IndexError:
                pass
        bad_names = [c for c in g['configs'] if '__' in c]
        if bad_names:
            problems.append(f"'__' in config name(s) {bad_names}: collect.py would "
                            f'misparse the filename')
        if problems:
            ok = False
            print(f'  FAIL {name}: ' + '; '.join(problems))
        else:
            print(f'  ok   {name}: {len(got)} units, bijection onto '
                  f'{len(g["configs"])} configs x {len(g["seeds"])} seeds x '
                  f'{len(g["months"])} months')
    return ok


def main():
    import sys
    if '--verify' in sys.argv:
        raise SystemExit(0 if verify() else 1)
    for name in GROUPS:
        g = group(name)
        n = n_units(name)
        print(f'\n=== group {name}: {g["note"]}')
        print(f'    configs {len(g["configs"])}  seeds {list(g["seeds"])}  '
              f'months {list(g["months"])}  fingerprint {registry_fingerprint(name)}')
        print(f'    numeric designs to cache: {needed_missing_modes(name)}')
        print(f'    prep : sbatch --array=0-{len(g["months"]) - 1} '
              f'cluster/nnsweep_array.sbatch <campaign> fixed {name}')
        print(f'    sweep: sbatch --array=0-{n - 1}%8 '
              f'cluster/nnsweep_array.sbatch <campaign> sweep {name}')
    print('\n--- stage A arms ---')
    for c in STAGE_A:
        d = CONFIGS[c]
        print(f'  {c:9s} ep{d["epochs"]:<3d} b{d["batch"]:<6d} lr{d["lr"]:<7.1e} '
              f'wd{d["weight_decay"]:<7.1e} drop{d["dropout"]:<5.2f} '
              f'cap{d["emb_cap"]:<4d} {str(d["hidden"]):<22s} {d["schedule"]:<8s} '
              f'{d["loss"]:<5s} {d["missing"]:<8s} {d.get("note", "")}')
    print()
    verify()


if __name__ == '__main__':
    main()
