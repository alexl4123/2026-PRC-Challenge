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
--pool V1,V2,... (2026-09-28, post-hoc arm M): seq = the plain mean of all listed variants' finals (3 seeds each, equal
weight per fit, exactly as seq_swap_eval pools the folds); weights from the pool's swap json; every finals folder must
match its fold campaign's _config.json.
    python experiments/exp81_submit_v21.py --pool ep30,ep40,ep50,ep60,ep80,ep40_d128,ep50_d128,ep40_l3,ep40_lr3 --version 24
--pool-seeds N (2026-09-28, arm N): each pooled variant contributes seeds 0..N-1 of its finals (default 3); the swap json
is then the one seq_swap_eval writes for the spec "seqsw_<v>:0,..,N-1+...", i.e. swap_seqsw_<v1>_s0..N-1__seqsw_<v2>_s...
    python experiments/exp81_submit_v21.py --pool <same 9> --pool-seeds 6 --check v21 --prev 24
--member-seeds 6 --weights-json members_s012345__seqS (2026-09-28, arm O): the 7 adsb2 members' finals use seeds 0-5 and the
weights come from experiments/seq/member_seeds_eval.py (must be PASS).
--arm-u U|UT (2026-10-02; experiments/seq/arm_u_eval.py, frozen gates, must be PASS): routes A/B/C outside P get
per-bin weights, bins = terciles of the per-row sd across the 27 pool finals (fixed 2025 cut points); UT adds
lambda_bin * (1 - p) * (seq10 - seqnc10) from cache/seq_final/ (seqnc finals needed). Route D / P as v24.
Label-free gates checked here: 2026 A/B/C bin shares within 10 points of 33%, mean shift per airport x month in [-6, +6] s.
    python experiments/exp81_submit_v21.py --pool <9> --arm-u U --check v21 --prev 24
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
    ap.add_argument('--pool', default=None, help='comma-separated variant names whose finals are averaged (arm M)')
    ap.add_argument('--pool-seeds', type=int, default=3, help='seeds 0..N-1 of each pooled variant (arm N: 6)')
    ap.add_argument('--member-seeds', type=int, default=3, help='seeds 0..N-1 of the 7 adsb2 members (arm O: 6)')
    ap.add_argument('--arm-u', default=None, choices=['U', 'UT'], help='disagreement-tercile A/B/C weights (cache/seq/arm_u.json)')
    ap.add_argument('--weights-json', default=None, help='cache/seq/<name>.json to take weights_new from (arm O: members_s012345__seqS)')
    a = ap.parse_args()
    pool = a.pool.split(',') if a.pool else None
    name = a.variant or f'ep{a.epochs}'
    if pool:
        tag = '' if a.pool_seeds == 3 else '_s' + ''.join(str(i) for i in range(a.pool_seeds))
        name = '__'.join(f'seqsw_{v}{tag}' for v in pool)[len('seqsw_'):]   # swap json: swap_seqsw_<v1>[_s..]__seqsw_<v2>...
    if a.check is None and a.version is None:
        fail('pass --version N or --check identity|v21')
    ident = a.check == 'identity'
    SEQF = CACHE / ('seq_final' if ident else f'seq_{name}_final')
    if ident:
        W = json.load(open(CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'))['weights']
    else:
        S = json.load(open(CACHE / 'seq' / (f'{a.weights_json}.json' if a.weights_json else f'swap_seqsw_{name}.json')))
        if S['verdict'] != 'PASS' or not all(S['rule'].values()):
            fail(f'swap eval did not pass: {S["verdict"]}')
        W = S['weights_new']
        for v in (pool or [name]):
            cf = json.load(open(CACHE / f'seq_{v}_final' / '_config.json'))
            cv = json.load(open(CACHE / 'cv' / f'seqsw_{v}' / '_config.json'))
            if cf != cv:
                fail(f'seq_{v}_final settings differ from the judged fold campaign seqsw_{v}: {cf} vs {cv}')
        if a.variant is None and pool is None and cf.get('epochs') != a.epochs:
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

    def mem(folder, f, n=3):
        if (folder, f, n) not in cache:
            files = [folder / f'{f}__seed{s:02d}.npy' for s in range(n)]
            miss = [str(p) for p in files + [folder / 'ranking_ids.npy'] if not p.exists()]
            if miss:
                fail(f'missing {miss}')
            if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
                fail(f'{folder} is in a different ranking order')
            cache[(folder, f, n)] = np.mean([np.load(p).astype(float) for p in files], axis=0)
        return cache[(folder, f, n)]

    seq_cache = {}

    def seq_arr():                                  # the seq member's ranking prediction (one variant, or the pool)
        if 'x' not in seq_cache:
            seq_cache['x'] = (np.mean([mem(CACHE / f'seq_{v}_final', 'seq', a.pool_seeds) for v in pool], axis=0) if pool
                              else mem(SEQF, 'seq'))
        return seq_cache['x']

    fam = lambda f: seq_arr() if f == 'seq' else mem(FA, f, a.member_seeds)

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
    gate_fail = []
    if a.arm_u:
        J = json.load(open(CACHE / 'seq' / 'arm_u.json'))
        if J[a.arm_u]['verdict'] != 'PASS':
            fail(f'arm {a.arm_u} did not pass its frozen gates')
        POOLU = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
        if pool != POOLU or a.pool_seeds != 3 or a.member_seeds != 3:
            fail('--arm-u is defined on the 27-fit pool (9 settings x seeds 0-2) with members at seeds 0-2')
        fits = np.array([np.load(CACHE / f'seq_{v}_final' / f'seq__seed{k:02d}.npy').astype(float) for v in POOLU for k in range(3)])
        sdr = fits.std(0); del fits
        bin_ = np.digitize(sdr, np.array(J['cuts']))
        if a.arm_u == 'UT':
            dlp = np.mean([(1 - pt[k]) for k in range(pt.shape[0])], axis=0) * (mem(CACHE / 'seq_final', 'seq') - mem(CACHE / 'seq_final', 'seqnc'))
        shares = [float(np.mean(bin_[ABC] == c)) for c in (0, 1, 2)]
        print(f'arm {a.arm_u}: cut points {J["cuts"]}, 2026 A/B/C bin shares ' + ' '.join(f'{x:.3f}' for x in shares))
        if any(abs(x - 1 / 3) > 0.10 for x in shares):
            gate_fail.append(f'bin shares {shares}')
        for c in (0, 1, 2):
            sel = ABC & (bin_ == c)
            val = blend(J[a.arm_u]['weights'][str(c)])[sel]
            if a.arm_u == 'UT':
                val = val + J[a.arm_u]['lambdas'][str(c)] * dlp[sel]
            new[sel] = np.rint(np.clip(val, 0, 200000))
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]; p14 = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(in_P)):
        fail('the TabICL probabilities do not cover the P rows exactly once')
    mix = lambda pb: np.clip(p14 * et[rows] + (1 - p14) * pb, 0, None)
    w2mix = mem(W2F, 'mix')[rows]
    new[rows] = np.rint(np.clip(v19[rows] + mix(0.5 * w2mix + 0.5 * seq_arr()[rows])
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
    sq, sq10 = seq_arr(), mem(CACHE / 'seq_final', 'seq')
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
    pv = t.pivot_table(index='apt', columns='month', values='d', aggfunc='mean')
    print(pv.round(2).to_string())
    if a.arm_u and (pv.abs().to_numpy() > 6).any():
        gate_fail.append(f'airport x month mean shift beyond +-6 s: max {pv.abs().to_numpy().max():.2f}')
    if gate_fail:
        fail('label-free gate(s) failed: ' + '; '.join(gate_fail))
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
