"""Bet D, step 1 evaluation: does the neighbour context carry signal on a held-out month? (reads s01 outputs)

All numbers on the held-out month's BULK rows (y <= 7200) of the subset airports -- the rows the bulk regressor
is responsible for. Reported:
  1. RMSE of each arm (per seed and seed-mean), of the v17-structure OOF (cache/w2/v17s_oof.npy) and of the seven
     adsb2 members' bulk regressors (cache/cv/adsb2/*__mMM.bulk.npy, seeds 0-2 mean) on the same rows.
  2. ctx - noctx (seed-mean vs seed-mean) with an airport-day block bootstrap CI, per airport, and the seed-to-seed
     spread of each arm (s0 vs s1) as the noise yardstick.
  3. Blend value: v17 + w (arm - v17), w fitted on half of the airport-days and scored on the other half (2-fold
     cross-fit, 20 random day splits); for ctx, noctx and both together. Context-specific value = ctx vs noctx.
  4. Residual correlation of each arm with v17 and with every member's bulk regressor.
  5. Residual arms (y - v17 target): ctx_res vs noctx_res vs v17.
  6. LABEL-FREE row selection (added after the LIRF subset): sections 1-5 select rows by y <= 7200, which conditions
     on the label and penalises v17's tail mixture (a bulk model then 'gains' by undoing it -- meaningless at Rome).
     Section 6 repeats the blend checks on ALL held-out rows outside zone P (P = no NM record and off_sched > 2 h,
     label-free), with v17 + w (ctx - v17) and v17 + w (ctx - noctx) cross-fitted as before.

    python experiments/seq/s02_subset_eval.py --tag LTFM-LFPG-EHAM__tr050608__te07__K64
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE, TAIL_CAP

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3']


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def block_boot(y, a, b, block, n_boot=1000, seed=0):
    """RMSE(a) - RMSE(b) and a 95% CI resampling whole blocks (airport x day)."""
    da, db = (y - a) ** 2, (y - b) ** 2
    _, inv = np.unique(block, return_inverse=True)
    SA, SB, N = np.bincount(inv, da), np.bincount(inv, db), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); nb = len(N)
    out = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        n = w @ N
        out[i] = np.sqrt(w @ SA / n) - np.sqrt(w @ SB / n)
    return rmse(y, a) - rmse(y, b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def crossfit(y, base, arms, block, n_split=20):
    """RMSE of base + sum_k w_k (arm_k - base), w by least squares on half the blocks, scored on the other half."""
    ub = np.unique(block)
    rng = np.random.default_rng(0)
    res = []
    for s in range(n_split):
        half = np.isin(block, rng.permutation(ub)[: len(ub) // 2])
        pred = base.copy()
        for fit, app in ((half, ~half), (~half, half)):
            D = np.column_stack([a[fit] - base[fit] for a in arms])
            w = np.linalg.lstsq(D, y[fit] - base[fit], rcond=None)[0]
            pred[app] = base[app] + np.column_stack([a[app] - base[app] for a in arms]) @ w
        res.append(rmse(y, pred) - rmse(y, base))
    return float(np.mean(res)), float(np.min(res)), float(np.max(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='LTFM-LFPG-EHAM__tr050608__te07__K64')
    a = ap.parse_args()
    d = CACHE / 'seq' / 'subset' / a.tag
    R = np.load(d / 'rows.npz', allow_pickle=True)
    va, y_all, fr = R['va'], R['y'], R['frame_row']
    y, apt, v17 = y_all[va], R['apt'][va], R['v17'][va]
    test_month = int(np.unique(R['month'][va])[0])
    k = y <= TAIL_CAP
    arms = {}
    for f in sorted(d.glob('*__s*.npy')):
        arm, s = f.stem.split('__s')
        arms.setdefault(arm, {})[int(s)] = np.load(f).astype(float)
    print(f'{a.tag}: held-out month {test_month}, rows {va.sum():,}, bulk rows {k.sum():,}; arms '
          + ', '.join(f'{m} seeds {sorted(v)}' for m, v in arms.items()))

    # members' bulk regressors on the same rows (their month-mm fold file covers every row of that month)
    base = np.load(CACHE / 'v11' / 'base.npz'); month_full = base['month']
    pos = np.searchsorted(np.flatnonzero(month_full == test_month), fr[va])
    mem = {}
    for fam in FAMS:
        fs = [CACHE / 'cv' / 'adsb2' / f'{fam}__seed{s:02d}__m{test_month:02d}.bulk.npy' for s in range(3)]
        if all(f.exists() for f in fs):
            mem[fam] = np.mean([np.load(f).astype(float)[pos] for f in fs], axis=0)
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    ids = dd.MVT_ID_mvt.to_numpy()[fr[va]]
    assert np.array_equal(ids, R['ids'][va]), 'row ids misaligned'
    day = dd.MVT_TIME_UTC_mvt.dt.strftime('%m%d').to_numpy()[fr[va]]
    block = np.char.add(apt.astype(str), day.astype(str))
    ad = pd.read_parquet(CACHE / 'adsb_lol' / 'features_training.parquet', columns=['MVT_ID_mvt', 'adsb_off_best'])
    own_adsb = ad.set_index('MVT_ID_mvt').reindex(ids).adsb_off_best.notna().to_numpy()

    yk, bk, ak = y[k], block[k], apt[k]
    mean_arm = {m: np.mean([v[s] for s in sorted(v)], axis=0) for m, v in arms.items()}

    print('\n1. RMSE on held-out bulk rows (all subset airports | per airport)')
    apts = sorted(set(ak))
    rows = [('v17 structure OOF', v17)] + [(f'member {f} (bulk, 3 seeds)', p) for f, p in mem.items()]
    for m, v in arms.items():
        for s in sorted(v):
            rows.append((f'{m} seed {s}', v[s]))
        if len(v) > 1:
            rows.append((f'{m} seed-mean', mean_arm[m]))
    print(f"  {'':34s}{'all':>8s}" + ''.join(f'{x:>8s}' for x in apts))
    for nm, p in rows:
        print(f'  {nm:34s}{rmse(yk, p[k]):8.2f}' + ''.join(f'{rmse(yk[ak == x], p[k][ak == x]):8.2f}' for x in apts))

    if 'ctx' in arms and 'noctx' in arms:
        print('\n2. context effect: ctx - noctx (seed-mean each), airport-day block bootstrap 95% CI')
        c, n = mean_arm['ctx'][k], mean_arm['noctx'][k]
        print(f'  all: {"%+.2f [%+.2f, %+.2f]" % block_boot(yk, c, n, bk)}')
        for x in apts:
            m = ak == x
            print(f'  {x}: {"%+.2f [%+.2f, %+.2f]" % block_boot(yk[m], c[m], n[m], bk[m])}')
        own = own_adsb[k]
        for nm, m in (('target has own ADS-B off-block', own), ('target without', ~own)):
            if m.sum() > 100:
                print(f'  {nm} (n={m.sum():,}): {rmse(yk[m], c[m]) - rmse(yk[m], n[m]):+.2f}')
        for m in ('ctx', 'noctx'):
            if len(arms[m]) > 1:
                s = sorted(arms[m])
                print(f'  seed spread {m}: s{s[0]} - s{s[1]} = {rmse(yk, arms[m][s[0]][k]) - rmse(yk, arms[m][s[1]][k]):+.2f}')
        s0 = sorted(set(arms['ctx']) & set(arms['noctx']))
        for s in s0:
            print(f'  seed-matched ctx - noctx, seed {s}: {rmse(yk, arms["ctx"][s][k]) - rmse(yk, arms["noctx"][s][k]):+.2f}')

    print('\n3. blend value on top of the v17 OOF (2-fold airport-day cross-fit, mean [min, max] of 20 splits)')
    for nm, lst in (('+ ctx', ['ctx']), ('+ noctx', ['noctx']), ('+ ctx + noctx', ['ctx', 'noctx']),
                    ('+ ctx_res', ['ctx_res']), ('+ noctx_res', ['noctx_res'])):
        if all(m in mean_arm for m in lst):
            print(f'  v17 {nm:16s}: {"%+.2f [%+.2f, %+.2f]" % crossfit(yk, v17[k], [mean_arm[m][k] for m in lst], bk)}')
    for x in apts:
        m = ak == x
        if 'ctx' in mean_arm and 'noctx' in mean_arm:
            print(f'  {x}: + ctx {crossfit(yk[m], v17[k][m], [mean_arm["ctx"][k][m]], bk[m])[0]:+.2f}   '
                  f'+ noctx {crossfit(yk[m], v17[k][m], [mean_arm["noctx"][k][m]], bk[m])[0]:+.2f}')

    print('\n4. residual correlation on held-out bulk rows')
    refs = [('v17', v17)] + list(mem.items())
    print(f"  {'':12s}" + ''.join(f'{r:>9s}' for r, _ in refs))
    for m in mean_arm:
        e = yk - mean_arm[m][k]
        print(f'  {m:12s}' + ''.join(f'{np.corrcoef(e, yk - p[k])[0, 1]:9.3f}' for _, p in refs))
    e17 = yk - v17[k]
    print('  (members among themselves, vs v17: ' + ', '.join(f'{f} {np.corrcoef(e17, yk - p[k])[0, 1]:.3f}' for f, p in mem.items()) + ')')

    label_free(R, mean_arm, v17, va, block)

    if 'ctx_res' in mean_arm and 'noctx_res' in mean_arm:
        print('\n5. residual fits on y - v17 (their prediction = v17 + net)')
        for m in ('ctx_res', 'noctx_res'):
            print(f'  {m}: {"%+.2f [%+.2f, %+.2f]" % block_boot(yk, mean_arm[m][k], v17[k], bk)} vs v17')
        print(f'  ctx_res - noctx_res: {"%+.2f [%+.2f, %+.2f]" % block_boot(yk, mean_arm["ctx_res"][k], mean_arm["noctx_res"][k], bk)}')


def label_free(R, arms_mean, v17_all, va, block_all):
    base = np.load(CACHE / 'v11' / 'base.npz')
    P = base['P'][R['frame_row']][va]
    y = R['y'][va]
    m = ~P
    ym, vm, bm, am = y[m], v17_all[m], block_all[m], R['apt'][va][m]
    print(f'\n6. label-free rows: held-out rows outside zone P = {m.sum():,} of {len(m):,} (all labels; y > 7200 kept: '
          f'{int((ym > TAIL_CAP).sum())})')
    print(f'  v17 RMSE {rmse(ym, vm):.2f}')
    have = [x for x in ('ctx', 'noctx') if x in arms_mean]
    for nm, lst in (('+ w (ctx - v17)', [arms_mean['ctx'][m]] if 'ctx' in have else None),
                    ('+ w (noctx - v17)', [arms_mean['noctx'][m]] if 'noctx' in have else None)):
        if lst is not None:
            print(f'  v17 {nm:22s}: {"%+.2f [%+.2f, %+.2f]" % crossfit(ym, vm, lst, bm)} | ' + '  '.join(
                f'{x} {crossfit(ym[am == x], vm[am == x], [l[am == x] for l in lst], bm[am == x])[0]:+.2f}'
                for x in sorted(set(am))))
    if len(have) == 2:
        dl = vm + (arms_mean['ctx'][m] - arms_mean['noctx'][m])      # base + delta: crossfit fits w on the delta
        print(f'  v17 {"+ w (ctx - noctx)":22s}: {"%+.2f [%+.2f, %+.2f]" % crossfit(ym, vm, [dl], bm)} | ' + '  '.join(
            f'{x} {crossfit(ym[am == x], vm[am == x], [dl[am == x]], bm[am == x])[0]:+.2f}' for x in sorted(set(am))))


if __name__ == '__main__':
    main()
