"""EXP81 -- v21 = v20 with the seq member retrained at 15 epochs, swapped in everywhere v20 uses seq.

DECISION (2026-09-27): experiments/seq/seq_swap_eval.py --new seqsw_ep15 PASSED its pre-registered rule R1-R5
(held-out -0.65 [-0.81,-0.52], scored -0.62 [-0.95,-0.37], Jan -0.35, Jul -0.89, 12/12 months, every seed and
every airport better). Weights: cache/seq/swap_seqsw_ep15.json ('weights_new').

Assembly (both parts from components, so no clip artefact can be carried forward):
  outside P   A/B/C: rint(clip(blend8)), D: rint(clip(blend8 + 0.5 * corr_old))   (exp79, seq = cache/seq_ep15_final/)
  zone P      rint(clip(v19[P] + mix(p, 0.5 W2 mix + 0.5 seq bulk) - mix(p, 0.5 W2 mix + 0.5 W2 nn)))
              v19 is the last version whose P rows are all unclipped (asserted), so v19[P] - mix(nn branch) is the exact
              non-branch part (corrector, stack); v20 has one P row at the 0 clip and cannot serve as the base.

    python experiments/exp81_submit_v21.py --check identity   # seq = 10-epoch finals + shipped weights -> must equal v20
    python experiments/exp81_submit_v21.py --check v21        # change summary, no file written
    python experiments/exp81_submit_v21.py --version 21

GENERALISED 2026-09-27 for v22 (seq at 20 epochs; experiments/seq/seq_swap_eval.py --old seqsw_ep15 --new seqsw_ep20
PASSED, arm B of the pre-registered decision tree): --epochs N selects cache/seq_epN_final/ and cache/seq/swap_seqsw_epN.json;
--prev V is the version the summary / rebuild compares with.
    python experiments/exp81_submit_v21.py --epochs 15 --check rebuild --prev 21   # must equal the shipped v21 exactly
    python experiments/exp81_submit_v21.py --epochs 20 --check v21 --prev 21       # v22 change summary vs v21
    python experiments/exp81_submit_v21.py --epochs 20 --version 22
--variant NAME (2026-09-27 overnight grid, e.g. ep60, ep40_d128) uses cache/seq_<NAME>_final/ and
cache/seq/swap_seqsw_<NAME>.json instead, and requires the finals' _config.json to equal the fold campaign's
(cache/cv/seqsw_<NAME>/_config.json) -- the ranking predictions come from exactly the settings that were judged.
    python experiments/exp81_submit_v21.py --variant ep30 --check rebuild --prev 23   # must equal the shipped v23
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load, peak_rss_gb
from prc.submit import build, TEAM

BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3', 'seq')
FA, W2F = CACHE / 'adsb2_final', CACHE / 'w2_final'


def fail(msg):
    sys.exit('FATAL: ' + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', choices=['identity', 'v21', 'rebuild'], default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--epochs', type=int, default=15, help='seq variant: cache/seq_ep<N>_final + swap_seqsw_ep<N>.json')
    ap.add_argument('--prev', type=int, default=20, help='version to compare with (summary / rebuild)')
    ap.add_argument('--variant', default=None, help='seq variant name (overrides --epochs): ep60, ep40_d128, ...')
    a = ap.parse_args()
    name = a.variant or f'ep{a.epochs}'
    if a.check is None and a.version is None:
        fail('pass --version N or --check identity|v21')
    ident = a.check == 'identity'
    SEQF = CACHE / ('seq_final' if ident else f'seq_{name}_final')
    if ident:
        W = json.load(open(CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'))['weights']
    else:
        S = json.load(open(CACHE / 'seq' / f'swap_seqsw_{name}.json'))
        if S['verdict'] != 'PASS' or not all(S['rule'].values()):
            fail(f'swap eval did not pass: {S["verdict"]}')
        W = S['weights_new']
        cf = json.load(open(SEQF / '_config.json'))
        cv = json.load(open(CACHE / 'cv' / f'seqsw_{name}' / '_config.json'))
        if cf != cv:
            fail(f'{SEQF.name} settings differ from the judged fold campaign seqsw_{name}: {cf} vs {cv}')
        if a.variant is None and cf.get('epochs') != a.epochs:
            fail(f'{SEQF} is not the {a.epochs}-epoch run: {cf}')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pt, et, in_P = c['ids'], c['pt'], c['et'], c['in_P'].astype(bool)
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT', 'MVT_TIME_UTC_mvt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    ABC, DnP = ~D & ~in_P, D & ~in_P
    shipped = lambda v: pd.read_parquet(OUTPUTS / f'{TEAM}_v{v}.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
    v19, v20 = shipped(19), shipped(a.prev if not ident else 20)
    if np.isnan(v19).any() or np.isnan(v20).any():
        fail('v19 / the comparison version miss ranking ids')
    if (v19[in_P] <= 0).any():
        fail('v19 has P rows at the 0 clip -- it cannot serve as the P base')
    z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
    if not np.array_equal(z['ids'], ids):
        fail('m11f candidates in a different order')
    corr = np.clip(z['trailD_full'], 0, None) - z['v9']

    cache = {}

    def mem(folder, f):
        if (folder, f) not in cache:
            files = [folder / f'{f}__seed{s:02d}.npy' for s in range(3)]
            miss = [str(p) for p in files + [folder / 'ranking_ids.npy'] if not p.exists()]
            if miss:
                fail(f'missing {miss}')
            if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
                fail(f'{folder} is in a different ranking order')
            cache[(folder, f)] = np.mean([np.load(p).astype(float) for p in files], axis=0)
        return cache[(folder, f)]

    fam = lambda f: mem(SEQF if f == 'seq' else FA, f)

    def blend(Wr):
        out = []
        for s in range(pt.shape[0]):
            p = pt[s]; Wb = sum(Wr.get(f, 0.0) for f in BULK)
            o = Wb * p * et + (1 - p) * sum(Wr[f] * fam(f) for f in BULK if Wr.get(f, 0.0) > 0)
            if Wr.get('routed', 0.0) > 0:
                o = o + Wr['routed'] * fam('routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)

    new = np.full(len(ids), np.nan)
    new[ABC] = np.rint(np.clip(blend(W['routes A/B/C'])[ABC], 0, 200000))
    new[DnP] = np.rint(np.clip(blend(W['route D'])[DnP] + 0.5 * corr[DnP], 0, 200000))
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]; p14 = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(in_P)):
        fail('the TabICL probabilities do not cover the P rows exactly once')
    mix = lambda pb: np.clip(p14 * et[rows] + (1 - p14) * pb, 0, None)
    w2mix = mem(W2F, 'mix')[rows]
    new[rows] = np.rint(np.clip(v19[rows] + mix(0.5 * w2mix + 0.5 * mem(SEQF, 'seq')[rows])
                                - mix(0.5 * w2mix + 0.5 * mem(W2F, 'nn')[rows]), 0, 200000))
    if np.isnan(new).any():
        fail('some rows were not assembled')

    if ident:
        for nm, g in (('A/B/C outside P', ABC), ('D outside P', DnP), ('P', in_P)):
            dd = np.abs(new[g] - v20[g])
            print(f'{nm:16s} {g.sum():7,} rows: max |rebuild - v20| {dd.max():.0f} s, rows differing {int((dd > 0).sum())}')
        print('IDENTITY' if np.array_equal(new, v20) else 'NOT IDENTICAL', f'| peak RSS {peak_rss_gb():.2f} GB')
        return
    if a.check == 'rebuild':
        dd = np.abs(new - v20)
        print(f'REBUILD of v{a.prev} with the {name} seq: rows differing {int((dd > 0).sum())}, max {dd.max():.0f} s -> '
              + ('IDENTICAL' if np.array_equal(new, v20) else 'NOT IDENTICAL'))
        return
    d = new - v20
    sq, sq10 = mem(SEQF, 'seq'), mem(CACHE / 'seq_final', 'seq')
    print('weights: ' + ' | '.join(f'{k}: ' + ' '.join(f'{f} {w:.3f}' for f, w in W[k].items() if w) for k in W))
    print(f'seq {name} finals: mean {sq.mean():.1f} median {np.median(sq):.1f} max {sq.max():.0f}; corr with the '
          f'10-epoch finals {np.corrcoef(sq, sq10)[0, 1]:.3f} (mean {sq10.mean():.1f})')
    ch = d != 0
    print(f'rows changed vs v{a.prev}: {int(ch.sum()):,} (A/B/C {int((ch & ABC).sum()):,}, D outside P {int((ch & DnP).sum()):,}, '
          f'P {int((ch & in_P).sum()):,}); zeros v{a.prev} {int((v20 == 0).sum())} -> new {int((new == 0).sum())}')
    for nm, g in (('A/B/C outside P', ABC), ('D outside P', DnP), ('P', in_P)):
        print(f'  {nm:16s}: mean shift {d[g].mean():+.2f} s, |shift| median {np.median(np.abs(d[g])):.0f}, '
              f'p99 {np.percentile(np.abs(d[g]), 99):.0f}, max {d[g].max():+.0f} / {d[g].min():+.0f}')
    t = pd.DataFrame({'apt': rk['APT'].astype(str).to_numpy(), 'month': rk['MVT_TIME_UTC_mvt'].dt.month.to_numpy(), 'd': d})
    print(t.pivot_table(index='apt', columns='month', values='d', aggfunc='mean').round(2).to_string())
    if a.check:
        return
    out = OUTPUTS / f'{TEAM}_v{a.version}.parquet'
    if out.exists() and not a.overwrite:
        fail(f'{out.name} exists; pass --overwrite to replace it')
    path = build(ids, new, version=a.version, clip=(0.0, 200000.0))
    got = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
    if not np.array_equal(got, new):
        fail('the written file differs from the in-memory build')
    print(f'v{a.version} written and read back identical | peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
