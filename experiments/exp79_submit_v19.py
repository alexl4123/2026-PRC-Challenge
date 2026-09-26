"""EXP79 -- v19 = v18 with the 'seq' member (bet D, neighbour-context set model) as the 8th blend member.

DECISION (2026-09-26): experiments/seq/seq_eval.py --corr-scale 0.5 (the v18 structure) PASSED its pre-registered rule
R1-R5 on the primary arm (held-out -2.11 [-2.97,-1.53], scored -1.51 [-2.41,-0.89], Jan -1.00, Jul -2.01, 12/12
months, every seed alone better). The pre-registered LIRF-gated arm passed too but was weaker (-1.70), so by the
pre-registered tie-break the UNgated 8-member blend ships. The info arm (lambda * (seq - seqnc)) is not used.

WHAT CHANGES: routes A/B/C outside P and route D outside P, assembled from components (as exp78 did for route D, so no
delta-chain clip artefact can arise):
    A/B/C outside P:   rint(clip(blend8))
    D outside P:       rint(clip(blend8 + 0.5 * corr_old))            (corr_old as in exp78: the halved v10 corrector)
    blend8 = the production tail branch per LGB seed (exp77::blend) over the adsb2 finals (cache/adsb2_final/) plus the
             seq finals (cache/seq_final/seq__seed0{0,1,2}.npy = bulk), weights from
             cache/seq/seq_eval_seq_adsb2_corr0.5.json
Zone P: v18 unchanged.

    python experiments/exp79_submit_v19.py --check identity   # 7 members + adsb2 weights must reproduce v18
    python experiments/exp79_submit_v19.py --check v19        # change summary, no file written
    python experiments/exp79_submit_v19.py --version 19       # writes outputs/understanding-accordion_v19.parquet
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pathlib import Path
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load, peak_rss_gb
from prc.submit import build, TEAM

BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3', 'seq')
FOLDER = {'seq': CACHE / 'seq_final'}                    # every other family: cache/adsb2_final/
FA = CACHE / 'adsb2_final'
FACTOR = 0.5
SEQ_EVAL = CACHE / 'seq' / 'seq_eval_seq_adsb2_corr0.5.json'


def fail(msg):
    sys.exit('FATAL: ' + msg)


def need(*paths):
    miss = [str(p) for p in paths if not p.exists()]
    if miss:
        fail('missing inputs:\n  ' + '\n  '.join(miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', choices=['identity', 'v19'], default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--seq-folder', default=None, help='dry runs only: another folder with seq__seedNN.npy + ranking_ids.npy')
    a = ap.parse_args()
    if a.seq_folder:
        if a.version is not None:
            fail('--seq-folder is for dry runs (--check) only')
        FOLDER['seq'] = Path(a.seq_folder)
    if a.check is None and a.version is None:
        fail('pass --version N (explicit) or --check identity|v19')

    need(CACHE / 'final' / 'lgb_components.npz', CACHE / 'major' / 'm11f_candidates.npz',
         CACHE / 'w2' / 'adsb2_weights.json', SEQ_EVAL, OUTPUTS / f'{TEAM}_v18.parquet')
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pt, et, in_P = c['ids'], c['pt'], c['et'], c['in_P'].astype(bool)
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT', 'MVT_TIME_UTC_mvt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    ABC, DnP = ~D & ~in_P, D & ~in_P
    apt, month = rk['APT'].astype(str).to_numpy(), rk['MVT_TIME_UTC_mvt'].dt.month.to_numpy()
    v18 = pd.read_parquet(OUTPUTS / f'{TEAM}_v18.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids)
    if v18.isna().any():
        fail('v18 is missing ranking ids')
    v18 = v18.to_numpy().astype(float)

    z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
    if not np.array_equal(z['ids'], ids):
        fail('m11f candidates in a different order')
    corr = np.clip(z['trailD_full'], 0, None) - z['v9']
    if np.abs(corr[~D]).max() > 1e-9:
        fail('corr_old is nonzero on rows WITH an NM record')

    cache = {}

    def mem(f):
        if f not in cache:
            folder = FOLDER.get(f, FA)
            files = [folder / f'{f}__seed{s:02d}.npy' for s in range(3)]
            need(folder / 'ranking_ids.npy', *files)
            if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
                fail(f'{folder} is in a different ranking order')
            cache[f] = np.mean([np.load(p).astype(float) for p in files], axis=0)
            if not np.isfinite(cache[f]).all():
                fail(f'{f} finals contain non-finite values')
        return cache[f]

    def blend(W):                                      # exp77::blend, with seq as one more bulk family
        out = []
        for s in range(pt.shape[0]):
            p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            pbm = sum(W[f] * mem(f) for f in BULK if W.get(f, 0.0) > 0)
            o = Wb * p * et + (1 - p) * pbm
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * mem('routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)

    def assemble(W):
        new = v18.copy()
        new[ABC] = np.rint(np.clip(blend(W['routes A/B/C'])[ABC], 0, 200000))
        new[DnP] = np.rint(np.clip(blend(W['route D'])[DnP] + FACTOR * corr[DnP], 0, 200000))
        return new

    WA = json.load(open(CACHE / 'w2' / 'adsb2_weights.json'))
    S = json.load(open(SEQ_EVAL))
    if WA['seeds'] != [0, 1, 2]:
        fail('adsb2 weights are not the 3-seed weights')
    if not (S['verdict'].startswith('PASS') and all(S['rule'].values()) and S['seq_seeds'] == [0, 1, 2]
            and S['member_seeds'] == [0, 1, 2] and 'seq' in S['members']):
        fail(f'seq_eval json is not the 3-seed PASS: verdict {S["verdict"]}, rule {S["rule"]}, seq seeds '
             f'{S["seq_seeds"]}, member seeds {S["member_seeds"]}')
    for key in ('routes A/B/C', 'route D'):             # the "old" side of seq_eval must be what v17/v18 ship
        d = max(abs(S['weights_old'][key].get(f, 0.0) - WA['weights'][key].get(f, 0.0)) for f in WA['weights'][key])
        if d > 1e-4:
            fail(f'seq_eval old weights differ from cache/w2/adsb2_weights.json on {key} (max {d:.4f})')

    if a.check == 'identity':
        old = assemble(WA['weights'])
        for nm, g, tol in (('A/B/C outside P', ABC, 3.0), ('D outside P', DnP, 0.0)):
            dd = np.abs(old[g] - v18[g])
            print(f'{nm:16s} {g.sum():7,} rows: max |7-member rebuild - v18| {dd.max():.0f} s, rows differing '
                  f'{int((dd > 0).sum()):,} -> {"OK" if dd.max() <= tol else "MISMATCH"}')
        print(f'zone P {in_P.sum():,} rows: untouched by construction | peak RSS {peak_rss_gb():.2f} GB')
        return

    W8 = S['weights']
    new = assemble(W8)
    ch = new != v18
    print('weights: ' + ' | '.join(f'{k}: ' + ' '.join(f'{f} {w:.3f}' for f, w in W8[k].items() if w) for k in W8))
    sq = mem('seq')
    print(f'seq finals (bulk, ranking): mean {sq.mean():.1f} median {np.median(sq):.1f} min {sq.min():.0f} max '
          f'{sq.max():.0f}; corr with w3 finals {np.corrcoef(sq, mem("w3"))[0, 1]:.3f}, w3 mean {mem("w3").mean():.1f}')
    print(f'rows changed vs v18: {int(ch.sum()):,} (A/B/C outside P {int((ch & ABC).sum()):,}, D outside P '
          f'{int((ch & DnP).sum()):,}, inside P {int((ch & in_P).sum())})')
    d = new - v18
    for nm, g in (('A/B/C outside P', ABC), ('D outside P', DnP)):
        print(f'  {nm:16s}: mean shift {d[g].mean():+.2f} s, |shift| median {np.median(np.abs(d[g])):.0f}, '
              f'p99 {np.percentile(np.abs(d[g]), 99):.0f}, max {d[g].max():+.0f} / {d[g].min():+.0f}')
    t = pd.DataFrame({'apt': apt, 'month': month, 'd': d})[~in_P]
    print('mean shift by airport x month (outside P):')
    print(t.pivot_table(index='apt', columns='month', values='d', aggfunc='mean').round(2).to_string())
    if a.check == 'v19':
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
