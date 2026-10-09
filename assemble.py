"""Assemble the submission from model outputs only -- no earlier submission file is read.

Until v26 every submission was built as an EDIT of an earlier submission file (the delta chain v11 -> ... -> v26,
builders experiments/exp70-exp82). This script replaces that chain with one closed formula per row group, computed
from the stored model outputs ("components"). It is the canonical builder from 2026-10-02 on.

ROW GROUPS (ranking departures; route = whether the flight has an NM off-block record AOBT_3_flt)
  A/B/C   AOBT_3 present, not in zone P            rint(clip( blend_A ))
  D       AOBT_3 missing, not in zone P            rint(clip( blend_D + 0.5 * corr ))
  P       AOBT_3 missing, > 2 h behind schedule    rint(clip( p * off_sched + (1 - p) * pb + corr ))
blend_X = the production mixture of the 8 members averaged over the 5 production tail-classifier seeds:
          mean_s clip( Wb * p_s * et + (1 - p_s) * sum_f w_f * bulk_f + w_routed * routed ),  Wb = sum of bulk weights
members = 7 adsb2 families (cache/adsb2_final/, seeds 0-2, bulk predictions; 'routed' carries its own tail handling)
          + seq = mean of the 27-fit pool (9 settings x seeds 0-2, cache/seq_<V>_final/).
weights = per-route NNLS weights (cache/seq/<pool swap json>, 'weights_new'); with --arm-u U|UT, routes A/B/C use the
          per-bin weights of cache/seq/arm_u.json (bins = terciles of the per-row sd across the 27 pool fits, fixed
          2025 cut points), UT adds lambda_bin * mean_s(1 - p_s) * (seq10 - seqnc10).
corr    = the v10 route-D corrector, clip(trailD_full, 0) - v9 (cache/major/m11f_candidates.npz); zero where AOBT_3 exists.
p (P)   = TabICL tail probability: Rome own context (cache/strategy/tfm_P_lirf_ranking.npz), other airports pooled
          (cache/tfm_w4/pool_nonrome_ranking.npz);  pb = 0.5 * W2 'mix' bulk (cache/w2_final/) + 0.5 * seq pool mean.
p_s, et = production tail classifier probabilities (5 seeds) and the tail anchor (cache/final/lgb_components.npz).

EQUIVALENCE (--check): routes A/B/C and route D outside P must equal the shipped file exactly; zone P may differ by
the rounding the old chain accumulated (measured 2026-10-02: |diff| <= 1.8 s on all 1,720 P rows).

    python assemble.py --check 24                     # v24 design vs outputs/..._v24.parquet
    python assemble.py --arm-u U --check 26           # v26 design vs outputs/..._v26.parquet
    python assemble.py --arm-u UT --check 27          # v27 (built with this script)
    python assemble.py --arm-u UT --lobt-window --out v28

LOBT WINDOW (--lobt-window, v28; header written 2026-10-06 BEFORE the build but AFTER the numbers below were measured).
  Rule. In the 2025 training file every departure with an NM flight-plan match (LOBT_flt present: 2,062,577 rows, the same
  rows that carry AOBT_3_flt) has its recorded off-block within +-3606 s of LOBT: BLOCK - LOBT in [-3606, +3606] with ZERO
  exceptions, a sharp edge (215 rows in (3600, 3606], 45 in [-3606, -3600)), median +251 s. The extract's movement-to-flight
  join evidently required |BLOCK - LOBT| <= 3606 s. This is a property of how the dataset was built (an artefact of the join),
  not operational knowledge. Consequence for a row with LOBT: y = MVT - BLOCK lies in [MVT - LOBT - 3606, MVT - LOBT + 3606].
  Transformation. After rint, every prediction p of a row with LOBT becomes
      q = min(max(p, ceil(max(0, off_lobt - 3606))), floor(off_lobt + 3606)),   off_lobt = MVT - LOBT (s).
  No row ids, no margin, no other change. In 2026 LOBT exists on exactly the 339,551 rows with AOBT_3 (routes A/B/C), so
  route D and zone P never move. If the rule holds in 2026, every moved row's error can only shrink (projection onto an
  interval that contains the truth).
  Evidence (2025, in-sample for the rule): on the v24 OOF 236 rows change, 0 get worse; held-out -0.550 [-0.946, -0.239],
  Jan+Jul -1.070 [-2.324, -0.308], 12/12 months better; ~79% of the gain sits in 10 rows (concentration disclosed; the
  "not carried by 100 rows" check is vacuous for a change that cannot worsen any row).
  2026 (label-free): 63 rows of v27 lie outside their window (60 below, 3 above); bound: RMSE <= 237.16 if the rule holds;
  ~96% of the bound is one LIRF July row (v27 52,348 s, window [83,214, 90,426]). Label-free consistency 2025 vs 2026: LOBT
  coverage 98.92% / 98.47%, MVT - LOBT > 3606 s on 2.22% / 2.11% of rows, min MVT - LOBT -3,382 / -3,173 s.
  Credit. The idea was adopted on 2026-10-06 from public competitor repositories (EnioAguiar/prc-taxiout-2026,
  javidmardanov/PRC-Data-Challenge-2026, joris-decombe/taxiout); the code here is our own. Competitors' leaderboard
  statements are NOT used as evidence. Pre-commitment: upload once; no later upload changes a projected row; nobody derives
  an implied label from the resulting score.

JOIN COMPLEMENT + SOURCE CAP (--join-complement, v29; header written 2026-10-06 BEFORE the build, AFTER the rules were
discovered and measured on 2025 by the 10-06 structure hunt -- not pre-registered in the strict sense; disclosed as such).
  Clean link. For a departure d WITHOUT LOBT, IFR, ADES != own airport: the arrival r (same file, ARR phase) with the
  smallest MVT_r - MVT_d in (0, 86400) s, APT_r = ADES_d, ADEP_r = APT_d, FLIGHT_d == FLIGHT_r or FLIGHT_d == CALLSIGN_r,
  LOBT present on r, and r's FLIGHT_ID appearing on no departure row. lo/hi = (MVT_d - LOBT_r) -+ 3606.
  A (join complement). d carries no NM record although its flight's NM record exists on the arrival side, so d's own join
    failed: |BLOCK_d - LOBT_r| > 3606 (2025: 0 exceptions on 1,426 clean links). Where lo < -12 (y below lo impossible,
    min recorded y is -12), y > hi: q = max(p, ceil(hi)).  2025: 70/70 one-sided rows above hi.
  B' (two-sided, lower side). Where lo >= -12, (MVT_d - SCHED_d) lies in [lo, hi] (a block stamped at the schedule would
    have joined) and the take-off does not cross local midnight relative to the schedule (dloc <= 0, guarding Rome's
    day-shift class): q = min(p, floor(max(lo, 0))).  EMPIRICAL: 2025 1,296/1,296 below lo, 0 above hi -- not a forced
    identity.
  C (source cap). LEBL and LEMD record no departure taxi above 3,606 s (2025: 0 of 391,947; max 3,604 / 3,598; other
    airports exceed it; 2026 arrivals at LEBL/LEMD also stay <= 3,606): q = min(q, 3606).
  Order: v28 (incl. the LOBT window) -> A -> B' -> C. Must not move any of the 63 rows the LOBT window projected (asserted).
  2026 label-free checks (abort if violated): clean-link share of IFR no-LOBT departures in [5%, 10%]; arrival NM coverage
  >= 99%; no moved row on an airport-day whose IFR no-LOBT share exceeds 17.2% (2025 max).  Expected: 36 rows move
  (9 up by A, 21 down by B', 6 down by C); bound if the rules hold: <= 235.023 from v28's 235.3996 (label-free moves).
  ~60% of the squared moves sit on one Rome row. Pre-commitment: upload once, score logged only, no implied label, no later
  upload changes these rows.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc.config import CACHE, OUTPUTS, ID
from prc.io import load

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
BULK = ['mix', 'nn', 'perapt', 'lintree', 'xgboost', 'w3', 'seq']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
TEAM = 'understanding-accordion'
HI = 200000.0


def fail(msg):
    sys.exit('FATAL: ' + msg)


class Components:
    """Loads every model output once, checks that all share the ranking row order."""

    def __init__(self):
        c = np.load(CACHE / 'final' / 'lgb_components.npz', allow_pickle=True)
        self.ids, self.pt, self.et, self.in_P = c['ids'], c['pt'], c['et'], c['in_P'].astype(bool)
        rk = load('ranking', phase='DEP', columns=[ID, 'AOBT_3_flt', 'APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'])
        if not np.array_equal(rk[ID].to_numpy(), self.ids):
            fail('ranking order differs from cache/final/lgb_components.npz')
        self.apt = rk.APT.astype(str).to_numpy(); self.month = rk.MVT_TIME_UTC_mvt.dt.month.to_numpy()
        self.off_sched = (rk.MVT_TIME_UTC_mvt - rk.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
        self.off_lobt = (rk.MVT_TIME_UTC_mvt - rk.LOBT_flt).dt.total_seconds().to_numpy()
        D = rk.AOBT_3_flt.isna().to_numpy()
        self.ABC, self.DnP = ~D & ~self.in_P, D & ~self.in_P
        z = np.load(CACHE / 'major' / 'm11f_candidates.npz', allow_pickle=True)
        if not np.array_equal(z['ids'], self.ids):
            fail('m11f candidates in a different order')
        self.corr = np.clip(z['trailD_full'], 0, None) - z['v9']
        if np.abs(self.corr[~D]).max() > 1e-9:
            fail('the route-D corrector is nonzero on a row with an NM record')
        self._cache = {}

    def fit(self, folder, fam, seeds=(0, 1, 2)):
        """Mean of the given seeds' predictions of one family in one finals folder."""
        k = (folder, fam, tuple(seeds))
        if k not in self._cache:
            f = CACHE / folder
            files = [f / f'{fam}__seed{s:02d}.npy' for s in seeds]
            miss = [str(p) for p in files + [f / 'ranking_ids.npy'] if not p.exists()]
            if miss:
                fail(f'missing {miss}')
            if not np.array_equal(np.load(f / 'ranking_ids.npy', allow_pickle=True), self.ids):
                fail(f'{folder} is in a different ranking order')
            arr = np.mean([np.load(p).astype(float) for p in files], axis=0)
            if not np.isfinite(arr).all():
                fail(f'{folder}/{fam} has non-finite predictions')
            self._cache[k] = arr
        return self._cache[k]

    def member(self, f):
        return np.mean([self.fit(f'seq_{v}_final', 'seq') for v in POOL], axis=0) if f == 'seq' else self.fit('adsb2_final', f)

    def blend(self, W):
        out = []
        for s in range(self.pt.shape[0]):
            p = self.pt[s]; Wb = sum(W.get(f, 0.0) for f in BULK)
            o = Wb * p * self.et + (1 - p) * sum(W[f] * self.member(f) for f in BULK if W.get(f, 0.0) > 0)
            if W.get('routed', 0.0) > 0:
                o = o + W['routed'] * self.member('routed')
            out.append(np.clip(o, 0, None))
        return np.mean(out, axis=0)


WINDOW = 3606


def lobt_selftest():
    """The window must hold exactly on the 2025 training departures (guards against unit / timezone bugs)."""
    d = load('training', phase='DEP', columns=['BLOCK_TIME_UTC_mvt', 'LOBT_flt'])
    x = (d.BLOCK_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().dropna().to_numpy()
    if len(x) < 2_000_000 or np.abs(x).max() != WINDOW:
        fail(f'LOBT self-test failed: {len(x):,} rows, max |BLOCK - LOBT| {np.abs(x).max():.0f} s (expected exactly {WINDOW})')
    print(f'LOBT self-test: {len(x):,} training departures, max |BLOCK - LOBT| = {WINDOW} s, 0 outside -> OK')


def lobt_window(C, pred):
    m = np.isfinite(C.off_lobt)
    if (m & ~C.ABC).any():
        fail('LOBT present on a route-D or zone-P row -- the window design assumed A/B/C only')
    lo = np.ceil(np.maximum(0.0, C.off_lobt[m] - WINDOW)); hi = np.floor(C.off_lobt[m] + WINDOW)
    q = pred.copy(); q[m] = np.minimum(np.maximum(pred[m], lo), hi)
    return q


TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam',
      'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome', 'LSZH': 'Europe/Zurich',
      'LTFM': 'Europe/Istanbul'}
CAP_APTS, CAP = ('LEBL', 'LEMD'), 3606
JC_COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt',
           'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'CALLSIGN_flt', 'APT']


def _secs(x):
    return (x - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def clean_links(which):
    """Departure rows (DEP order of `which`) with a clean link -> (row index, lo, hi, off_sched, dloc)."""
    f = load(which, columns=JC_COLS + (['BLOCK_TIME_UTC_mvt'] if which == 'training' else []))
    dep = f[f.PHASE_mvt.eq('DEP')].reset_index(drop=True); arr = f[f.PHASE_mvt.eq('ARR')].reset_index(drop=True)
    for x in (dep, arr):
        for c in ('APT', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'CALLSIGN_flt'):
            x[c] = x[c].astype(str)
    dep['i'] = np.arange(len(dep))
    used = set(dep.FLIGHT_ID_mvt.dropna().unique())
    D = dep[dep.LOBT_flt.isna() & dep.FLIGHT_RULE_mvt.astype(str).eq('I') & (dep.APT != dep.ADES_mvt)].copy()
    D['MVT'] = _secs(D.MVT_TIME_UTC_mvt); D['SCHED'] = _secs(D.SCHED_TIME_UTC_mvt)
    R = arr[arr.LOBT_flt.notna() & ~arr.FLIGHT_ID_mvt.isin(used)].copy()
    R = pd.DataFrame({'a_FLIGHT_mvt': R.FLIGHT_mvt, 'a_CALLSIGN_flt': R.CALLSIGN_flt, 'a_ADEP_mvt': R.ADEP_mvt, 'a_APT': R.APT,
                      'a_MVT': _secs(R.MVT_TIME_UTC_mvt), 'a_LOBT': _secs(R.LOBT_flt)})
    L = []
    for key in ('a_FLIGHT_mvt', 'a_CALLSIGN_flt'):
        m = D.merge(R, left_on=['FLIGHT_mvt', 'APT', 'ADES_mvt'], right_on=[key, 'a_ADEP_mvt', 'a_APT'])
        m['t'] = m.a_MVT - m.MVT; L.append(m[(m.t > 0) & (m.t < 86400)])
    m = pd.concat(L).sort_values('t').drop_duplicates('i').reset_index(drop=True)
    off = (m.MVT - m.a_LOBT).to_numpy()
    dloc = np.zeros(len(m), int)
    for a, z in TZ.items():
        k = (m.APT == a).to_numpy()
        if k.any():
            mv = pd.to_datetime(m.MVT[k] - 900, unit='s', utc=True).dt.tz_convert(z).dt.tz_localize(None).dt.normalize()
            sc = pd.to_datetime(m.SCHED[k], unit='s', utc=True).dt.tz_convert(z).dt.tz_localize(None).dt.normalize()
            dloc[k] = (mv - sc).dt.days.to_numpy()
    out = dict(i=m.i.to_numpy(), lo=off - WINDOW, hi=off + WINDOW, off_sched=(m.MVT - m.SCHED).to_numpy(), dloc=dloc,
               n_ifr_nolobt=len(D), dep=dep)
    if which == 'training':
        out['y'] = (m.MVT - _secs(m.BLOCK_TIME_UTC_mvt)).to_numpy()
    return out


def jc_selftest():
    """Re-run the 2025 premises of A, B' and C on the training file; abort if any exception appears."""
    L = clean_links('training'); y, lo, hi = L['y'], L['lo'], L['hi']
    inside = (y >= lo) & (y <= hi)
    one = lo < -12; b = ~one & (L['off_sched'] >= lo) & (L['off_sched'] <= hi) & (L['dloc'] <= 0)
    d = load('training', phase='DEP', columns=['APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'])
    yd = (d.MVT_TIME_UTC_mvt - d.BLOCK_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    cap_viol = int((yd[d.APT.astype(str).isin(CAP_APTS).to_numpy()] > CAP).sum())
    msg = (f'join-complement self-test (2025): clean links {len(y):,}, inside +-3606 {int(inside.sum())}; one-sided {int(one.sum())}, '
           f'below hi {int((y[one] <= hi[one]).sum())}; B\' rows {int(b.sum())}, above lo {int((y[b] >= lo[b]).sum())}; '
           f'LEBL/LEMD above {CAP}: {cap_viol}')
    if inside.any() or (y[one] <= hi[one]).any() or (y[b] >= lo[b]).any() or cap_viol or len(y) < 1000:
        fail(msg)
    print(msg + ' -> OK')


def join_complement(C, pred, before_window):
    """Apply A, B' and C to the v28 prediction (after the LOBT window)."""
    L = clean_links('ranking'); dep = L['dep']
    if not np.array_equal(dep.MVT_ID_mvt.to_numpy(), C.ids):
        fail('ranking DEP order differs from the components')
    i, lo, hi, osd, dloc = L['i'], L['lo'], L['hi'], L['off_sched'], L['dloc']
    share = len(i) / max(L['n_ifr_nolobt'], 1)
    arr = load('ranking', phase='ARR', columns=['LOBT_flt', 'FLIGHT_RULE_mvt'])
    cov = arr[arr.FLIGHT_RULE_mvt.astype(str).eq('I')].LOBT_flt.notna().mean()
    q = pred.copy()
    one = lo < -12
    q[i[one]] = np.maximum(q[i[one]], np.ceil(hi[one]))
    b = ~one & (osd >= lo) & (osd <= hi) & (dloc <= 0)
    q[i[b]] = np.minimum(q[i[b]], np.floor(np.maximum(lo[b], 0)))
    cap = np.isin(C.apt, CAP_APTS)
    m = np.isfinite(C.off_lobt) & cap
    if (np.ceil(np.maximum(0.0, C.off_lobt[m] - WINDOW)) > CAP).any():
        fail('the LEBL/LEMD cap conflicts with a LOBT window lower edge')
    q[cap] = np.minimum(q[cap], CAP)
    moved = q != pred
    if (moved & (before_window != pred)).any():
        fail('the join complement would move a row the LOBT window projected (pre-commitment)')
    day = pd.to_datetime(dep.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d').to_numpy()
    ifr_nolobt = (dep.LOBT_flt.isna() & dep.FLIGHT_RULE_mvt.astype(str).eq('I')).to_numpy()
    sh = pd.Series(ifr_nolobt).groupby([dep.APT.to_numpy(), day]).mean()
    worst = max((sh.get((a, dd), 0.0) for a, dd in zip(dep.APT.to_numpy()[moved], day[moved])), default=0.0)
    print(f'join complement 2026: clean links {len(i)} ({share:.1%} of {L["n_ifr_nolobt"]:,} IFR no-LOBT deps), arrival IFR NM coverage '
          f'{cov:.2%}; moved: A {int((q[i[one]] != pred[i[one]]).sum())}, B\' {int((q[i[b]] != pred[i[b]]).sum())}, '
          f'C {int((cap & moved).sum())}; total {int(moved.sum())}; worst airport-day IFR no-LOBT share among moved rows {worst:.1%}')
    if not (0.05 <= share <= 0.10) or cov < 0.99 or worst > 0.172:
        fail('a 2026 label-free consistency check failed')
    return q


def assemble(C, arm=None, lobt=False, jc=False):
    name = '__'.join(f'seqsw_{v}' for v in POOL)
    S = json.load(open(CACHE / 'seq' / f'swap_{name}.json'))
    if S['verdict'] != 'PASS':
        fail('the pool weights do not come from a passing evaluation')
    W = S['weights_new']
    pred = np.full(len(C.ids), np.nan)
    pred[C.ABC] = C.blend(W['routes A/B/C'])[C.ABC]
    if arm:
        J = json.load(open(CACHE / 'seq' / 'arm_u.json'))
        if J[arm]['verdict'] != 'PASS':
            fail(f'arm {arm} did not pass its frozen gates')
        fits = np.array([C.fit(f'seq_{v}_final', 'seq', (k,)) for v in POOL for k in range(3)])
        b = np.digitize(fits.std(0), np.array(J['cuts'])); del fits
        if arm == 'UT':
            dl = np.mean(1 - C.pt, axis=0) * (C.fit('seq_final', 'seq') - C.fit('seq_final', 'seqnc'))
        for c in (0, 1, 2):
            sel = C.ABC & (b == c)
            v = C.blend(J[arm]['weights'][str(c)])[sel]
            pred[sel] = v + (J[arm]['lambdas'][str(c)] * dl[sel] if arm == 'UT' else 0.0)
    pred[C.DnP] = (C.blend(W['route D']) + 0.5 * C.corr)[C.DnP]
    t1 = np.load(CACHE / 'strategy' / 'tfm_P_lirf_ranking.npz', allow_pickle=True)
    t2 = np.load(CACHE / 'tfm_w4' / 'pool_nonrome_ranking.npz', allow_pickle=True)
    rows = np.r_[t1['pos'], t2['pos']]; p = np.r_[t1['p_tabicl'], t2['p_pool']].astype(float)
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(C.in_P)):
        fail('the TabICL probabilities do not cover the zone-P rows exactly once')
    pb = 0.5 * C.fit('w2_final', 'mix')[rows] + 0.5 * C.member('seq')[rows]
    pred[rows] = p * C.off_sched[rows] + (1 - p) * pb + C.corr[rows]
    if np.isnan(pred).any():
        fail('some rows were not assembled')
    out = np.rint(np.clip(pred, 0, HI))
    if not lobt:
        return out
    win = lobt_window(C, out)
    return join_complement(C, win, out) if jc else win


def shipped(C, v):
    s = pd.read_parquet(OUTPUTS / f'{TEAM}_v{v}.parquet').set_index(ID)['TAXITIME_SEC_mvt'].reindex(C.ids)
    if s.isna().any():
        fail(f'v{v} misses ranking ids')
    return s.to_numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm-u', default=None, choices=['U', 'UT'])
    ap.add_argument('--check', type=int, default=None, help='compare with outputs/..._v<N>.parquet')
    ap.add_argument('--out', default=None, help='write outputs/understanding-accordion_<out>.parquet (e.g. v27)')
    ap.add_argument('--lobt-window', action='store_true', help='project A/B/C predictions into the LOBT +-3606 s window (v28)')
    ap.add_argument('--join-complement', action='store_true', help="v29: join-complement rules A, B' and the LEBL/LEMD cap C (needs --lobt-window)")
    a = ap.parse_args()
    C = Components()
    if a.lobt_window:
        lobt_selftest()
    if a.join_complement:
        if not a.lobt_window:
            fail('--join-complement is defined on top of --lobt-window')
        jc_selftest()
    new = assemble(C, a.arm_u, a.lobt_window, a.join_complement)
    if a.check is not None:
        old = shipped(C, a.check)
        for nm, g, tol in (('A/B/C outside P', C.ABC, 0.0), ('D outside P', C.DnP, 0.0), ('zone P', C.in_P, 2.0)):
            d = np.abs(new[g] - old[g])
            print(f'{nm:16s} {g.sum():7,} rows: rows differing {int((d > 0).sum()):5,}, max |diff| {d.max():4.0f} s '
                  f'-> {"OK" if d.max() <= tol else "MISMATCH"}')
        ok = all(np.abs(new[g] - old[g]).max() <= t for g, t in ((C.ABC, 0), (C.DnP, 0), (C.in_P, 2)))
        print(f'v{a.check}: ' + ('EQUIVALENT (P differs by old-chain rounding only)' if ok else 'NOT EQUIVALENT'))
        if not ok:
            sys.exit(1)
    if a.out:
        from prc.submit import build
        out = OUTPUTS / f'{TEAM}_{a.out}.parquet'
        if out.exists():
            fail(f'{out.name} exists; choose another name')
        path = build(C.ids, new, version='final' if a.out == 'final' else int(a.out.lstrip('v')), clip=(0.0, HI))
        got = pd.read_parquet(path).set_index(ID)['TAXITIME_SEC_mvt'].reindex(C.ids).to_numpy().astype(float)
        if not np.array_equal(got, new):
            fail('the written file differs from the in-memory build')
        print(f'{out.name} written and read back identical')


if __name__ == '__main__':
    main()
