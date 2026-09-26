"""EXP78 -- v18 = v17 with the route-D corrector HALVED on route D outside P, assembled from components
(which also repairs the delta-chain clip artefact on those rows).

DECISION (user, 2026-09-26; evidence reports/2026-09-26_routeD_corrector.md, candidate "c half"):
  route D outside P (3,570 ranking rows: no NM record = AOBT_3 missing, and not in P) serve
        v18 = rint(clip(blendD + FACTOR * corr_old, 0, 200000)),      FACTOR = 0.5
    blendD   = the adsb2 7-member route-D blend exactly as exp77 --arm adsb2 builds it (cache/adsb2_final/ seeds 0-2,
               weights cache/w2/adsb2_weights.json 'route D', production tail branch per LGB seed)
    corr_old = clip(m11f trailD_full, 0) - m11f v9   (the v10 corrector, carried additively to v17)
  Zone P, routes A/B/C and every other row: the v17 integers, untouched.

WHY FROM COMPONENTS, NOT A DELTA ON v17. v17 on these rows is the end of the delta chain
    v11 = rint(clip(v10f + stack))  ->  v15 = rint(clip(v14 + exp75 delta))  ->  v16 (exp76)  ->  v17 (exp77)
(v12..v14 do not touch these rows). Where an intermediate value hit the 0 clip (v11: the stack correction, once at
v15), the next delta subtracted the UNclipped term, so v17 != blendD + corr_old there: the "chain-bug" rows.
A delta on v17 would carry that artefact forward and would add a new one wherever blendD + corr_old < 0 (v17 is 0
there). Assembling from components removes both. The price is that v17 also carries one rint per chain step, so the
formula at factor 1.0 differs from v17 by 1-2 s on ~1,600 rows. That is rounding only, and --check identity proves it
with an exact replay of the chain from the same components.

CHAIN-BUG ROWS are defined by mechanism, not by a threshold: the route-D-outside-P rows whose v17 integer changes when
the intermediate 0-clips are removed from the exact replay. (r00_arrays.py's 45-row set used |v17 - (blendD +
corr_old)| > 2 s; --check identity prints how the two sets differ.)

    python experiments/exp78_submit_v18.py --check identity   # (1) factor 1.0, no fix
    python experiments/exp78_submit_v18.py --check fix        # (2) factor 1.0 with the fix
    python experiments/exp78_submit_v18.py --check v18        # (3) factor 0.5 + fix, change summary, no file written
    python experiments/exp78_submit_v18.py --version 18       # writes outputs/understanding-accordion_v18.parquet
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load, peak_rss_gb
from prc.submit import build, TEAM

BULK = ('mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3')
FACTOR = 0.5
LO, HI = 0.0, 200000.0
R00 = CACHE / 'routed_0926' / 'arrays_2026.npz'


def fail(msg):
    sys.exit('FATAL: ' + msg)


def need(*paths):
    miss = [str(p) for p in paths if not p.exists()]
    if miss:
        fail('missing inputs:\n  ' + '\n  '.join(miss))


def r(x):
    """What prc.submit.build does to a prediction: clip, then round half to even."""
    return np.rint(np.clip(x, LO, HI))


def shipped(v, ids):
    f = OUTPUTS / f'{TEAM}_v{v}.parquet'
    need(f)
    s = pd.read_parquet(f).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids)
    if s.isna().any():
        fail(f'{f.name} is missing {int(s.isna().sum())} ranking ids')
    return s.to_numpy().astype(float)


def components():
    need(CACHE / 'final' / 'lgb_components.npz', CACHE / 'major' / 'm11f_candidates.npz',
         CACHE / 'final' / 'stack_c_ranking.npz', CACHE / 'w2' / 'adsb2_weights.json',
         CACHE / 'w2' / 'w2_weights.json', CACHE / 'w2' / 'w3_weights.json')
    c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
    ids, pb, pt, et, in_P = c['ids'], c['pb'], c['pt'], c['et'], c['in_P'].astype(bool)
    rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT', 'MVT_TIME_UTC_mvt'])
    if not np.array_equal(rk[ID].to_numpy(), ids):
        fail('ranking order differs from the components')
    D = rk['AOBT_3_flt'].isna().to_numpy()
    DnP = D & ~in_P
    cache = {}

    def mem(folder, f):
        k = (folder, f)
        if k not in cache:
            files = [folder / f'{f}__seed{s:02d}.npy' for s in range(3)]
            need(folder / 'ranking_ids.npy', *files)
            if not np.array_equal(np.load(folder / 'ranking_ids.npy', allow_pickle=True), ids):
                fail(f'{folder} is in a different ranking order')
            cache[k] = np.mean([np.load(f).astype(float) for f in files], axis=0)
        return cache[k]

    def blend(folder, W):                   # exp77::blend, verbatim logic
        out = []
        for s in range(pt.shape[0]):
            p = pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            pbm = sum(W[f] * mem(folder, f) for f in BULK if W.get(f, 0.0) > 0)
            o = Wb * p * et + (1 - p) * pbm
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * mem(folder, 'routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)

    WA = json.load(open(CACHE / 'w2' / 'adsb2_weights.json'))
    W2 = json.load(open(CACHE / 'w2' / 'w2_weights.json'))
    if WA['seeds'] != [0, 1, 2] or W2['seeds'] != [0, 1, 2]:
        fail('adsb2 / w2 weights are not the 3-seed weights')
    W3 = json.load(open(CACHE / 'w2' / 'w3_weights.json'))
    F2, FA = CACHE / 'w2_final', CACHE / 'adsb2_final'
    blendD = blend(FA, WA['weights']['route D'])

    z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
    if not np.array_equal(z['ids'], ids):
        fail('m11f candidates in a different order')
    v10f = np.clip(z['trailD_full'], 0, None)
    corr = v10f - z['v9']
    if np.abs(corr[~D]).max() > 1e-9:
        fail('corr_old is nonzero on rows WITH an NM record')

    # pieces of the shipped delta chain on route D outside P (exp70, exp75, exp76, exp77 --arm adsb2)
    st = np.load(CACHE / 'final' / 'stack_c_ranking.npz', allow_pickle=True)
    if not (np.array_equal(st['ids'], ids) and np.array_equal(st['in_P'].astype(bool), in_P)):
        fail('stack_c_ranking in a different order or with a different P')
    stk = np.where(in_P, 0.0, st['corr'].astype(float))
    need(CACHE / 'nn_final' / 'ranking_ids.npy')
    if not np.array_equal(np.load(CACHE / 'nn_final' / 'ranking_ids.npy', allow_pickle=True), ids):
        fail('nn_final is in a different ranking order')
    net = np.mean([np.load(CACHE / 'nn_final' / f'nn_final_seed{s:02d}.npy').astype(float) for s in range(3)], axis=0)
    v9D = np.mean([np.clip(pt[s] * et + (1 - pt[s]) * (0.5 * pb[s] + 0.5 * net), 0, None) for s in range(pb.shape[0])], axis=0)
    b6 = blend(F2, W2['weights']['route D'])
    b7 = blend(F2, W3['weights']['route D'])
    return dict(ids=ids, in_P=in_P, D=D, DnP=DnP, ABC=~D & ~in_P, apt=rk.APT.astype(str).to_numpy(),
                month=rk.MVT_TIME_UTC_mvt.dt.month.to_numpy(), blendD=blendD, corr=corr,
                v11f=v10f + stk, a=b6 - v9D - stk, b=b7 - b6, c=blendD - b7)


def chain(C, v17):
    """Exact replay of v11 -> v15 -> v16 -> v17 on route D outside P, and the same replay without the
    intermediate 0-clips. Returns (replayed v17, chain-bug mask, stage at which a clip engaged)."""
    DnP = C['DnP']
    s11 = C['v11f']; v11 = r(s11)
    s15 = v11 + C['a']; v15 = r(s15)
    s16 = v15 + C['b']; v16 = r(s16)
    v17r = r(v16 + C['c'])
    u = np.rint(np.rint(np.rint(s11) + C['a']) + C['b'])
    u17 = r(u + C['c'])                      # only the final clip, which is intended
    bug = DnP & (u17 != v17r)
    stage = np.where(s11 < 0, 'v11', np.where(s15 < 0, 'v15', np.where(s16 < 0, 'v16', '-')))
    return v17r, v11, v15, v16, bug, stage


def assemble(C, v17, bug, factor, fix):
    """Float prediction handed to prc.submit.build."""
    pred = v17.copy()
    DnP = C['DnP']
    pred[DnP] = np.clip(C['blendD'] + factor * C['corr'], LO, HI)[DnP]
    if not fix:                              # keep the chain's value on the bug rows, moved by the design delta only
        pred[bug] = np.clip(v17 + (factor - 1.0) * C['corr'], LO, HI)[bug]
    return pred


def confinement(C, new, v17, label):
    ch = new != v17
    print(f'  {label}: rows changed vs v17 {int(ch.sum()):,} | route D outside P {int((ch & C["DnP"]).sum()):,} | '
          f'zone P {int((ch & C["in_P"]).sum())} | routes A/B/C {int((ch & C["ABC"]).sum())} | '
          f'outside route-D-outside-P {int((ch & ~C["DnP"]).sum())}')
    if (ch & ~C['DnP']).any():
        fail('a row outside route D outside P changed')
    return ch


def bug_table(C, v17, new, bug, stage, r00bad):
    k = np.flatnonzero(bug | r00bad)
    t = pd.DataFrame({'row': k, 'MVT_ID': C['ids'][k].astype(np.int64), 'apt': C['apt'][k], 'month': C['month'][k],
                      'mechanism': bug[k], 'in_r00_45': r00bad[k], 'clip_at': stage[k],
                      'blendD': C['blendD'][k].round(1), 'corr_old': C['corr'][k].round(1),
                      'intended': np.clip(C['blendD'] + C['corr'], LO, HI)[k].round(1),
                      'v17_old': v17[k].astype(int), 'new': new[k].astype(int), 'new-old': (new - v17)[k].astype(int)})
    return t.sort_values(['apt', 'row'])


def shift_summary(C, v17, new):
    DnP = C['DnP']; s = new - v17; ch = s != 0
    print(f'  route D outside P ({DnP.sum():,} rows): changed {int(ch[DnP].sum()):,} | |shift| > 2 s: '
          f'{int((np.abs(s) > 2)[DnP].sum()):,} | > 60 s: {int((np.abs(s) > 60)[DnP].sum()):,} | '
          f'mean shift {s[DnP].mean():+.2f} s over all {DnP.sum():,} rows, {s[DnP & ch].mean():+.2f} s over changed rows')
    i = np.flatnonzero(DnP)[np.argmax(np.abs(s[DnP]))]
    print(f'  max |shift| {abs(s[i]):.0f} s: row {i} MVT_ID {int(C["ids"][i])} {C["apt"][i]} month {C["month"][i]} '
          f'{v17[i]:.0f} -> {new[i]:.0f} (blendD {C["blendD"][i]:.0f}, corr_old {C["corr"][i]:+.0f})')
    d = pd.DataFrame({'apt': C['apt'][DnP], 'month': C['month'][DnP], 's': s[DnP], 'ch': ch[DnP],
                      'corr': C['corr'][DnP]})
    g = lambda keys: d.groupby(keys).agg(n=('s', 'size'), changed=('ch', 'sum'), mean_shift=('s', 'mean'),
                                         mean_corr_old=('corr', 'mean'), min_shift=('s', 'min'), max_shift=('s', 'max'))
    pd.set_option('display.width', 200)
    print('  per month (route D outside P rows):')
    print(g('month').round(1).to_string().replace('\n', '\n    ').join(['    ', '']))
    print('  per airport:')
    print(g('apt').round(1).to_string().replace('\n', '\n    ').join(['    ', '']))
    print('  per airport x month:')
    print(g(['apt', 'month']).round(1).to_string().replace('\n', '\n    ').join(['    ', '']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=None)
    ap.add_argument('--check', choices=['identity', 'fix', 'v18'], default=None)
    ap.add_argument('--factor', type=float, default=FACTOR, help='corrector factor on route D outside P (decision: 0.5)')
    ap.add_argument('--overwrite', action='store_true', help='allow replacing an existing outputs/..._v<N>.parquet')
    a = ap.parse_args()
    if (a.check is None) == (a.version is None):
        fail('pass exactly one of --check {identity,fix,v18} or --version N (explicit)')
    if a.check in ('identity', 'fix'):
        a.factor = 1.0                       # checks 1 and 2 are defined at factor 1.0
    if a.version is not None:
        target = OUTPUTS / f'{TEAM}_v{a.version}.parquet'
        if target.exists() and not a.overwrite:
            fail(f'{target.name} exists; pass --overwrite to replace it')
        if a.factor != FACTOR:
            print(f'WARNING: building with factor {a.factor}, not the decided {FACTOR}')

    C = components()
    ids, DnP = C['ids'], C['DnP']
    v17 = shipped(17, ids)
    v11s, v14s, v15s, v16s = (shipped(v, ids) for v in (11, 14, 15, 16))
    v17r, v11r, v15r, v16r, bug, stage = chain(C, v17)
    r00bad = np.zeros(len(ids), bool)
    print(f'ranking rows {len(ids):,}; route D {C["D"].sum():,}, zone P {C["in_P"].sum():,}, route D outside P {DnP.sum():,}, '
          f'routes A/B/C outside P {C["ABC"].sum():,}')
    if R00.exists():
        z = np.load(R00, allow_pickle=True)
        if not np.array_equal(z['ids'], ids):
            fail('arrays_2026.npz in a different order')
        dB = np.abs(C['blendD'] - z['base'])[DnP].max(); dC = np.abs(C['corr'] - z['corr_old']).max()
        print(f'inputs vs {R00.name}: blendD max |d| {dB:.2e} s on route D outside P, corr_old max |d| {dC:.2e} s, '
              f'D / P masks equal: {np.array_equal(z["D"], C["D"])} / {np.array_equal(z["P"], C["in_P"])}')
        if dB > 1e-6 or dC > 1e-6:
            fail('blendD / corr_old differ from the report arrays')
        r00bad = z['chain_bad'].astype(bool)
    else:
        print(f'note: {R00} not found; skipping the cross-check against the report arrays')

    # replay identity is required in every mode: it is what proves the components are v17's
    rep = {'v11': int((v11r != v11s)[DnP].sum()), 'v14': int((v11r != v14s)[DnP].sum()),
           'v15': int((v15r != v15s)[DnP].sum()), 'v16': int((v16r != v16s)[DnP].sum()), 'v17': int((v17r != v17)[DnP].sum())}
    if any(rep.values()):
        fail(f'the exact chain replay does not reproduce the shipped files on route D outside P: {rep}')
    nb = int(bug.sum())
    print(f'chain replay from components reproduces shipped v11 / v14 / v15 / v16 / v17 on all {DnP.sum():,} route-D-outside-P '
          f'rows (differing: {rep}); chain-bug rows (by mechanism): {nb}')
    ones = lambda f, fx: r(assemble(C, v17, bug, f, fx))

    if a.check == 'identity':
        print(f'\n=== CHECK 1: factor 1.0, NO bug fix ===')
        print(f'  1a exact replay of the shipped chain from components: {rep["v17"]} of {DnP.sum():,} route-D-outside-P rows '
              f'differ from v17 -> IDENTITY')
        clip_rows = DnP & (stage != '-')
        print(f'  1b chain-bug rows = rows whose v17 changes when the intermediate 0-clips are removed from that replay: {nb} '
              f'| by airport ' + ', '.join(f'{k} {v}' for k, v in pd.Series(C['apt'][bug]).value_counts().items())
              + ' | clip engaged at ' + ', '.join(f'{k} {v}' for k, v in pd.Series(stage[bug]).value_counts().items())
              + f' | rows where a clip engaged but the final integer is unaffected: {int((clip_rows & ~bug).sum())}')
        if R00.exists():
            print(f'     vs r00_arrays.py chain_bad ({int(r00bad.sum())} rows, threshold |v17 - (blendD + corr_old)| > 2 s): '
                  f'common {int((bug & r00bad).sum())}, only r00 {int((r00bad & ~bug).sum())}, only here {int((bug & ~r00bad).sum())}')
        form = ones(1.0, True)               # the serving formula alone at factor 1.0
        d = form - v17
        print(f'  1c formula rint(clip(blendD + 1.0 * corr_old)) vs v17 on route D outside P: {int((d != 0)[DnP].sum()):,} rows differ; '
              f'|d| > 2 s on {int((np.abs(d) > 2)[DnP].sum())} rows, all chain-bug rows: {bool(np.all(bug[DnP & (np.abs(d) > 2)]))}; '
              f'on the other rows max |d| {np.abs(d[DnP & ~bug]).max():.0f} s '
              f'(|d|=1: {int((np.abs(d) == 1)[DnP & ~bug].sum()):,}, |d|=2: {int((np.abs(d) == 2)[DnP & ~bug].sum())}), '
              f'SSE {np.sum(d[DnP & ~bug] ** 2):.0f} s^2 vs the chain-bug SSE {np.sum(d[bug] ** 2):.3e} s^2')
        new = ones(1.0, False)
        print('  1d the build at factor 1.0 WITHOUT the fix (formula on route D outside P, chain-bug rows kept at v17):')
        ch = confinement(C, new, v17, 'build(1.0, no fix)')
        dd = new - v17
        print(f'     chain-bug rows differing from v17: {int((ch & bug).sum())}; every other change is rounding: max |d| '
              f'{np.abs(dd[ch]).max() if ch.any() else 0:.0f} s -> the components reproduce v17 up to the chain\'s per-step '
              f'rounding on every route-D-outside-P row except the {nb} chain-bug rows')
        worse = DnP & ~bug & (np.abs(r(np.clip(v17 + (0.5 - 1.0) * C['corr'], LO, HI)) - ones(0.5, True)) > 2)
        print(f'  (info) a delta build on v17 at factor 0.5 (v17 - 0.5 corr_old) would differ from the formula by > 2 s on '
              f'{int(worse.sum())} non-bug rows (rows where v17 sits on the 0 clip: blendD + corr_old < 0)')
    elif a.check == 'fix':
        print(f'\n=== CHECK 2: factor 1.0 WITH the bug fix ===')
        new, nofix = ones(1.0, True), ones(1.0, False)
        dif = new != nofix
        print(f'  build(1.0, fix) vs build(1.0, no fix): {int(dif.sum())} rows differ, all of them chain-bug rows: '
              f'{bool(np.all(bug[dif]))} (chain-bug rows the fix leaves unchanged: {int((bug & ~dif).sum())})')
        if not np.all(bug[dif]):
            fail('the fix moves a row that is not a chain-bug row')
        ch = confinement(C, new, v17, 'build(1.0, fix)')
        print(f'     of which chain-bug rows {int((ch & bug).sum())}; the other {int((ch & ~bug).sum()):,} are the rounding rows '
              f'of check 1 (max |d| {np.abs((new - v17)[ch & ~bug]).max():.0f} s)')
        s = new - v17
        print(f'  chain-bug repair: SSE of the change {np.sum(s[bug] ** 2):.3e} s^2, sum of the change {s[bug].sum():+.0f} s, '
              f'max |change| {np.abs(s[bug]).max():.0f} s')
        pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
        print('  chain-bug rows (mechanism) and r00 chain_bad rows; intended = clip(blendD + corr_old), new = rint(intended):')
        print(bug_table(C, v17, new, bug, stage, r00bad).to_string(index=False).replace('\n', '\n    ').join(['    ', '']))
    else:
        f = a.factor
        new = ones(f, True)
        if a.check == 'v18':
            print(f'\n=== CHECK 3: v18 = factor {f} + bug fix (no file written) ===')
        else:
            print(f'\n=== BUILD v{a.version}: factor {f} + bug fix ===')
        ch = confinement(C, new, v17, f'build({f}, fix)')
        print(f'  chain-bug rows changed {int((ch & bug).sum())} of {nb}; changed non-bug route-D-outside-P rows '
              f'{int((ch & ~bug).sum()):,}')
        shift_summary(C, v17, new)
        if a.version is not None:
            pred = assemble(C, v17, bug, f, True)
            path = build(ids, pred, version=a.version, clip=(LO, HI))
            got = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(ids).to_numpy().astype(float)
            print(f'  re-read {path.name}: equals the in-memory build on all rows: {bool(np.array_equal(got, new))}')
            if not np.array_equal(got, new):
                fail('the written file differs from the in-memory build')
            confinement(C, got, v17, f'written v{a.version}')
    print(f'\npeak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
