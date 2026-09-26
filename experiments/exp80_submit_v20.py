"""EXP80 -- v20 = v19 with the seq bulk in zone P's real-taxi branch (in place of the W2 network bulk).

DECISION (2026-09-26): experiments/seq/seq_p_branch.py PASSED its pre-registered rule R1-R5 (v19 structure, all
flights): held-out -0.65 [-1.29,-0.22], scored -0.73 [-1.28,-0.35], Jan -0.47, Jul -0.98, every seed alone better,
still < 0 without the 100 most-improved rows (-0.05 / -0.15: the gain is concentrated, as expected for 4,944 P rows).

Zone P (1,720 ranking rows) serves p * et + (1 - p) * pb (v13/v14 TabICL p, exp75). v15-v19:
        pb = 0.5 * W2 mix final + 0.5 * W2 nn final          (cache/w2_final/, seeds 0-2)
v20:    pb = 0.5 * W2 mix final + 0.5 * seq final bulk       (cache/seq_final/, seeds 0-2)
        delta = mix(p, pb_new) - mix(p, pb_old)   on the P rows only, added to v19 (exp75/exp77 pattern). v19 has no
        P row at the 0 clip (asserted), so v19 + delta is the exact component sum; the final clip in build() is then
        what a from-components assembly gives (1 row, whose route-D corrector exceeds the new mixture, goes to 0).
Every row outside P: v19 unchanged.

    python experiments/exp80_submit_v20.py --check      # pb_new := pb_old -> delta exactly 0
    python experiments/exp80_submit_v20.py --summary    # change summary, no file written
    python experiments/exp80_submit_v20.py --version 20
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load
from prc.submit import build, TEAM


def fail(msg):
    sys.exit('FATAL: ' + msg)


def seeds012(folder, fam, ids):
    files = [folder / f'{fam}__seed{s:02d}.npy' for s in range(3)]
    miss = [str(f) for f in files + [folder / 'ranking_ids.npy'] if not f.exists()]
    if miss:
        fail(f'missing: {miss}')
    if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
        fail(f'{folder} is in a different ranking order')
    return np.mean([np.load(f).astype(float) for f in files], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if not (a.check or a.summary) and a.version is None:
        fail('pass --version N, --check or --summary')

    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, et, in_P = c['ids'], c['et'], c['in_P'].astype(bool)
    rk = load('ranking', phase='DEP', columns=[ID, 'APT', 'MVT_TIME_UTC_mvt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('ranking order differs from the components')
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]
    p14 = np.r_[t1['p_tabicl'].astype(float), t2['p_pool'].astype(float)]
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(in_P)):
        fail('the TabICL probabilities do not cover the P rows exactly once')

    W2F = CACHE / 'w2_final'
    pb_old = 0.5 * seeds012(W2F, 'mix', ids)[rows] + 0.5 * seeds012(W2F, 'nn', ids)[rows]
    seqb = seeds012(CACHE / 'seq_final', 'seq', ids)[rows]
    pb_new = pb_old if a.check else 0.5 * seeds012(W2F, 'mix', ids)[rows] + 0.5 * seqb
    mix = lambda pb: np.clip(p14 * et[rows] + (1 - p14) * pb, 0, None)
    delta = np.zeros(len(ids))
    delta[rows] = mix(pb_new) - mix(pb_old)
    if a.check:
        print(f'CHECK (pb_new := pb_old): max |delta| {np.abs(delta).max():.2e} -> '
              f'{"IDENTITY" if np.abs(delta).max() < 1e-9 else "NOT ZERO"}')
        return

    v19 = pd.read_parquet(OUTPUTS / f'{TEAM}_v19.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids)
    if v19.isna().any():
        fail('v19 is missing ranking ids')
    v19 = v19.to_numpy().astype(float)
    # The delta is exact wherever v19 itself is unclipped (v19 = the unclipped component sum there); a row that was
    # already AT the 0 clip would carry the exp78 chain artefact, so that is fatal. Crossing 0 now is fine: build()
    # clips at 0, which is what a from-components assembly gives.
    if (v19[in_P] <= 0).any():
        fail(f'{int((v19[in_P] <= 0).sum())} P rows sit at the 0 clip in v19 -- a delta on v19 is not valid there')
    new = v19 + delta
    print(f'P rows where the final 0 clip engages (component sum < 0): {int((new[in_P] < 0).sum())}')
    apt, month = rk['APT'].astype(str).to_numpy(), rk['MVT_TIME_UTC_mvt'].dt.month.to_numpy()
    d = delta[rows]
    print(f'P rows {len(rows):,}: TabICL p mean {p14.mean():.3f}, (1-p) mean {(1 - p14).mean():.3f}')
    print(f'real-taxi branch: W2 nn bulk mean {(2 * pb_old - 2 * 0.5 * seeds012(W2F, "mix", ids)[rows]).mean():.0f} s, '
          f'seq bulk mean {seqb.mean():.0f} s; pb old mean {pb_old.mean():.0f} -> new {pb_new.mean():.0f}')
    print(f'delta on P: mean {d.mean():+.1f} s, |delta| median {np.median(np.abs(d)):.0f}, p99 '
          f'{np.percentile(np.abs(d), 99):.0f}, max {d.max():+.0f} / {d.min():+.0f}')
    t = pd.DataFrame({'apt': apt[rows], 'month': month[rows], 'd': d})
    print(t.pivot_table(index='apt', columns='month', values='d', aggfunc=['mean', 'count']).round(1).to_string())
    if a.summary:
        return
    out = OUTPUTS / f'{TEAM}_v{a.version}.parquet'
    if out.exists() and not a.overwrite:
        fail(f'{out.name} exists; pass --overwrite to replace it')
    path = build(ids, new, version=a.version, clip=(0.0, 200000.0))
    got = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
    if not np.array_equal(got, np.rint(np.clip(new, 0, 200000))):
        fail('the written file differs from the in-memory build')
    ch = got != v19
    print(f'v{a.version} vs v19: rows changed {int(ch.sum()):,} (outside P: {int((ch & ~in_P).sum())}); read back OK')


if __name__ == '__main__':
    main()
