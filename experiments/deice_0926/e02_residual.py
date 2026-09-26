"""De-icing / disruption indicator, step 2: residual harness on the v17-structure 2025 OOF.

PROTOCOL (the discord_eval harness, re-based on v17):
  base   = cache/w2/v17s_oof.npy (2,085,047 rows, load('training', phase='DEP') order; y, month, P, D from
           cache/v11/base.npz -- alignment asserted against cache/major/dep_training.parquet)
  target = y - base, fitted on rows outside zone P with y <= 7200 s
  model  = discord_eval/common.py::residual_arms: one LightGBM per arm (31 leaves, lr 0.05, 300 rounds),
           12 folds leave-one-month-out, the SAME fixed 500k-row subsample of each fold's training rows in
           every arm (paired arms); served = base + correction outside P, P keeps the base; scored on ALL
           flights, uncapped.
  CONTEXT (in every arm, = the CONTROL arm): APT, off_aobt3, off_sched, the v17 OOF prediction, local take-off
           hour; METAR at take-off (tmpc, dwpc, vsby, deice_likely, cold, frozen_precip, snow, freeze);
           EUROCONTROL daily arrival ATFM delay per flight (total and weather); and the flight's OWN adsb.lol
           summary (has_ground, off_best, best_src, stat_s, d_stand_first, off_first, stop_s, n_stops, hold_s,
           dwell_s, at_park, path_m, apt_day_ground, apt_day_best). So a candidate gain cannot come from
           re-reading the flight's own ADS-B, the weather flags or the ATFM day -- only from the aggregates.
  ARMS:
    control     context only
    cand        PRIMARY. + the 9 look-ahead ("whole") aggregates of e01: dz_ncov_h, dz_rr_sh_h, dz_rr_n_h,
                dz_ls_sh_h, dz_rest_h (other departures taking off within +-60 min), dz_ncov_d, dz_rr_sh_d,
                dz_rr_n_d, dz_ls_sh_d (other departures of the same local day)
    cand_past   SECONDARY. The same 9 columns built from flights that took off BEFORE ours only
                (trailing 60 min; day-so-far)
    placebo     the `cand` block permuted jointly across rows WITHIN airport (seed 0)
    cand_taxi   EXPLORATORY. cand + dz_taxi_h, dz_ntaxi_h (mean ADS-B taxi of others within +-60 min)
    cand_rr     EXPLORATORY. cand without the long-stop shares (dz_ls_sh_h, dz_ls_sh_d)

WHY THE WHOLE WINDOW IS PRIMARY (decided before any result): the organisers allow other rows and look-ahead
within the provided data, adsb.lol is approved open data, and no own label enters. Physically, departures that
take off in the hour AFTER ours were on the surface -- and in the same de-icing queue -- during our taxi-out,
so the centred window describes the conditions we taxied in; the trailing window sees only half of them. The
past-only arm is kept as the conservative variant in case look-ahead use is ever questioned.

PRE-REGISTERED ADOPTION RULE (written 2026-09-26 before any candidate result was computed):
  The PRIMARY arm `cand` is ADOPTED (i.e. recommended for an in-member retrain) only if ALL hold, with
  candidate - control differences in all-flights RMSE (negative = candidate better):
    R1  12-month difference < 0  AND  scored-month (Jan + Jul) difference < 0      (paired row bootstrap mean)
    R2  January alone < 0  AND  July alone < 0
    R3  airport-day BLOCK bootstrap (resample airport x local-day blocks, 1,000 draws) 95% CI for the 12-month
        difference entirely below 0 (upper bound < 0)
    R4  not carried by < 100 rows: after removing the 100 rows the candidate helped most, the 12-month AND the
        scored-month differences are still < 0
  VALIDITY: the candidate must beat the placebo on 12 months (cand - placebo < 0); if the placebo gains as much
  as the candidate, the verdict is "uninformative", whatever R1-R4 say.
  `cand_past` is judged by the same rule and reported; if only it passes, it is a follow-up candidate, not an
  adoption. `cand_taxi` / `cand_rr` are exploratory: reported, never adopted from this run.

Nothing is written to existing files. Per-arm predictions are cached in the session scratchpad; the context
frame and the results go to cache/deice_0926/.

    python experiments/deice_0926/e02_residual.py [--arms control cand cand_past placebo cand_taxi cand_rr]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault('DISCORD_SCRATCH', '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-'
                      'Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/deice_0926')
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'experiments', 'discord_eval'))
import numpy as np
import pandas as pd
from prc.config import CACHE
from prc.io import load, peak_rss_gb
from common import residual_arms, compare, fmt, rmse, SCRATCH      # discord_eval harness (unchanged)

OUT = CACHE / 'deice_0926'
CAND = ['dz_ncov_h', 'dz_rr_sh_h', 'dz_rr_n_h', 'dz_ls_sh_h', 'dz_rest_h', 'dz_ncov_d', 'dz_rr_sh_d', 'dz_rr_n_d',
        'dz_ls_sh_d']
CAND_PAST = [c.replace('dz_', 'dzp_') for c in CAND]
TAXI = ['dz_taxi_h', 'dz_ntaxi_h']
WX = ['wx_tmpc', 'wx_dwpc', 'wx_vsby', 'wx_deice_likely', 'wx_cold', 'wx_frozen_precip', 'wx_snow', 'wx_freeze']
ATFM = ['atfm_dly_per_flt', 'atfm_dly_W_per_flt']
OWN = ['adsb_has_ground', 'adsb_off_best', 'adsb_best_src', 'adsb_stat_s', 'adsb_d_stand_first', 'adsb_off_first',
       'adsb_stop_s', 'adsb_n_stops', 'adsb_hold_s', 'adsb_dwell_s', 'adsb_at_park', 'adsb_path_m',
       'adsb_apt_day_ground', 'adsb_apt_day_best']


def context_extra():
    """Weather, ATFM, own ADS-B and local hour, row-aligned with dep_training.parquet (cached)."""
    f = OUT / 'ctx_training.parquet'
    if f.exists():
        return pd.read_parquet(f)
    from prc.weather import download, prepare, attach as attach_wx
    from prc.disruption import attach as attach_atfm
    d = load('training', phase='DEP', columns=['MVT_ID_mvt', 'APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt']).reset_index(drop=True)
    ref = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt'])
    assert np.array_equal(d.MVT_ID_mvt.to_numpy(), ref.MVT_ID_mvt.to_numpy()), 'row order differs'
    wx = attach_wx(d, prepare(download()))
    at = attach_atfm(d)
    a = pd.read_parquet(CACHE / 'adsb_lol' / 'features_training.parquet', columns=['MVT_ID_mvt'] + OWN)
    a = a.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(d.MVT_ID_mvt.to_numpy())
    r = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['hour_loc', 'date_loc'])
    X = pd.DataFrame({'hour_loc': r.hour_loc.to_numpy(np.float32)})
    for c in WX:
        X[c] = wx[c].to_numpy(np.float32)
    for c in ATFM:
        X[c] = at[c].to_numpy(np.float32)
    for c in OWN:
        X[c] = a[c].to_numpy(np.float32)
    X['date_loc'] = r.date_loc.to_numpy()                 # block id only (dropped before modelling)
    X.to_parquet(f, index=False)
    return X


def load_B():
    b = np.load(CACHE / 'v11' / 'base.npz')
    y, month, P, D = b['y'], b['month'], b['P'], b['D']
    base = np.load(CACHE / 'w2' / 'v17s_oof.npy').astype(float)
    dd = pd.read_parquet(CACHE / 'major' / 'dep_training.parquet', columns=['MVT_ID_mvt', 'APT', 'off_sched', 'off_aobt3', 'y'])
    assert len(dd) == len(y) == len(base) == 2_085_047
    assert np.allclose(dd.y.to_numpy(float), y, equal_nan=True), 'dep_training not aligned with base.npz'
    ex = context_extra()
    ind = pd.read_parquet(OUT / 'ind_training.parquet')
    assert np.array_equal(ind.MVT_ID_mvt.to_numpy(), dd.MVT_ID_mvt.to_numpy()), 'indicator rows misaligned'
    ctx = pd.DataFrame({'APT': pd.Categorical(dd.APT.astype(str)), 'off_aobt3': dd.off_aobt3.to_numpy(float),
                        'off_sched': dd.off_sched.to_numpy(float), 'base': base})
    ctx = pd.concat([ctx, ex.drop(columns=['date_loc'])], axis=1)
    block = pd.factorize(dd.APT.astype(str) + '_' + ex.date_loc.astype(str))[0]
    return dict(y=y, month=month, P=P, D=D, base=base, ctx=ctx, ind=ind, block=block,
                apt=dd.APT.astype(str).to_numpy())


def block_bootstrap(y, a, b, block, n_boot=1000, seed=0):
    """RMSE(a) - RMSE(b), resampling whole blocks (airport x local day) with replacement."""
    da, db = (y - a) ** 2, (y - b) ** 2
    ids, inv = np.unique(block, return_inverse=True)
    SA = np.bincount(inv, da); SB = np.bincount(inv, db); N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); nb = len(ids)
    out = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        n = w @ N
        out[i] = np.sqrt(w @ SA / n) - np.sqrt(w @ SB / n)
    return float(np.sqrt(SA.sum() / N.sum()) - np.sqrt(SB.sum() / N.sum())), float(np.percentile(out, 2.5)), \
        float(np.percentile(out, 97.5)), int(nb)


def full_compare(B, a, b, n_boot_block=1000):
    res = compare(B, a, b)
    y, month, blk = B['y'], B['month'], B['block']
    for scope, k in (('12m', np.ones(len(y), bool)), ('scored', np.isin(month, [1, 7])), ('jan', month == 1),
                     ('jul', month == 7)):
        res[f'block_{scope}'] = block_bootstrap(y[k], a[k], b[k], blk[k], n_boot_block)
    res['per_month'] = [rmse(y[month == m], a[month == m]) - rmse(y[month == m], b[month == m]) for m in range(1, 13)]
    S = np.isin(month, [1, 7]) & (y <= 43_200)             # reporting only: scored months without > 12 h labels
    res['scored_le12h'] = rmse(y[S], a[S]) - rmse(y[S], b[S])
    return res


def verdict(r, plc=None):
    R1 = r['12m'][0] < 0 and r['scored'][0] < 0
    R2 = r['jan'][0] < 0 and r['jul'][0] < 0
    R3 = r['block_12m'][2] < 0
    R4 = r['drop100_12m'] < 0 and r['drop100_scored'] < 0
    v = dict(R1=bool(R1), R2=bool(R2), R3=bool(R3), R4=bool(R4))
    if plc is not None:
        v['valid_beats_placebo'] = bool(plc['12m'][0] < 0)
    v['ADOPT'] = bool(R1 and R2 and R3 and R4 and v.get('valid_beats_placebo', True))
    return v


def fmtb(r):
    f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
    return (f"BLOCK 12m {f(r['block_12m'])} | scored {f(r['block_scored'])} | Jan {f(r['block_jan'])} | "
            f"Jul {f(r['block_jul'])} | scored w/o y>12h {r['scored_le12h']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['control', 'cand', 'cand_past', 'placebo', 'cand_taxi', 'cand_rr'])
    a = ap.parse_args()
    t0 = time.time()
    B = load_B()
    print(f'loaded {time.time() - t0:.0f} s, ctx {B["ctx"].shape}, peak {peak_rss_gb():.2f} GB, scratch {SCRATCH}', flush=True)
    ind = B['ind']
    perm = np.arange(len(ind)); rng = np.random.default_rng(0)
    for ap_ in np.unique(B['apt']):
        k = np.flatnonzero(B['apt'] == ap_); perm[k] = rng.permutation(k)
    spec = {'control': None,
            'cand': ind[CAND],
            'cand_past': ind[CAND_PAST],
            'placebo': ind[CAND].iloc[perm].reset_index(drop=True),
            'cand_taxi': ind[CAND + TAXI],
            'cand_rr': ind[[c for c in CAND if '_ls_' not in c]]}
    arms = {k: spec[k] for k in (['control'] + [x for x in a.arms if x != 'control'])}
    preds = residual_arms(B, arms, tag='deice')
    y, month, base = B['y'], B['month'], B['base']
    S = np.isin(month, [1, 7])
    print(f'\nbase v17s: 12m {rmse(y, base):.3f}  scored {rmse(y[S], base[S]):.3f}  '
          f'(control: 12m {rmse(y, preds["control"]):.3f}, scored {rmse(y[S], preds["control"][S]):.3f})', flush=True)
    res = {}
    res['control_vs_base'] = full_compare(B, preds['control'], base)
    print(f"control vs base   : {fmt(res['control_vs_base'])}\n                    {fmtb(res['control_vs_base'])}", flush=True)
    for name in [x for x in arms if x != 'control']:
        r = full_compare(B, preds[name], preds['control'])
        res[name] = r
        print(f"{name:10s} - control: {fmt(r)}\n                    {fmtb(r)}\n"
              f"                    per month {' '.join(f'{v:+.2f}' for v in r['per_month'])}", flush=True)
    if 'placebo' in preds:
        for name in ('cand', 'cand_past'):
            if name in preds:
                res[f'{name}_vs_placebo'] = compare(B, preds[name], preds['placebo'])
                print(f"{name} - placebo: {fmt(res[f'{name}_vs_placebo'])}", flush=True)
    print('\nPRE-REGISTERED VERDICTS', flush=True)
    for name in ('cand', 'cand_past'):
        if name in res:
            v = verdict(res[name], res.get(f'{name}_vs_placebo'))
            res[f'verdict_{name}'] = v
            print(f'  {name}: {v}', flush=True)
    # where does the candidate move the error? per airport and by the hour's remote-rest share (label-free cut)
    rows = []
    for name in [x for x in ('cand', 'cand_past', 'cand_taxi') if x in preds]:
        p, c = preds[name], preds['control']
        for ap_ in np.unique(B['apt']):
            for scope, k in (('12m', B['apt'] == ap_), ('scored', (B['apt'] == ap_) & S), ('jan', (B['apt'] == ap_) & (month == 1))):
                rows.append(dict(arm=name, cut=f'apt={ap_}', scope=scope, n=int(k.sum()),
                                 d_rmse=rmse(y[k], p[k]) - rmse(y[k], c[k]),
                                 d_sse=float(((y[k] - p[k]) ** 2 - (y[k] - c[k]) ** 2).sum())))
        sh = ind.dz_rr_sh_h.to_numpy()
        for lab, k in (('rr_sh_h=NaN', np.isnan(sh)), ('rr_sh_h=0', sh == 0), ('0<rr_sh_h<=.05', (sh > 0) & (sh <= .05)),
                       ('.05<rr_sh_h<=.15', (sh > .05) & (sh <= .15)), ('rr_sh_h>.15', sh > .15)):
            rows.append(dict(arm=name, cut=lab, scope='12m', n=int(k.sum()), d_rmse=rmse(y[k], p[k]) - rmse(y[k], c[k]),
                             d_sse=float(((y[k] - p[k]) ** 2 - (y[k] - c[k]) ** 2).sum()),
                             rmse_control=rmse(y[k], c[k]), rmse_base=rmse(y[k], base[k])))
    cuts = pd.DataFrame(rows)
    cuts.to_csv(OUT / 'e02_cuts.csv', index=False)
    print('\nCUTS (candidate - control)\n' + cuts.round(3).to_string(), flush=True)
    json.dump(res, open(OUT / 'e02_results.json', 'w'), indent=1, default=float)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB, total {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
