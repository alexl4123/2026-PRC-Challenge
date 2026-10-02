"""RMSE-aligned tail specialist: weight the specialist's GBM by the squared-error cost of a
probability mistake, (off_sched - pb)^2, and/or use the continuous mixing target.
Replicates exp62 phase B (TE in-fold, same candidate cache); arms are PRE-REGISTERED below."""
import sys, time
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))))
from common import *
import lightgbm as lgb, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from prc.features import NEEDED, build as build_base, align_categories
from prc.models.tail_specialist import (TailSpecialist, TailSpecialistX, SPEC_FEATURES, SPEC_PARAMS, population)
from prc.p_features import te_features, FAMILIES
from prc.config import TAIL_CAP
t0 = time.time()
EXTRA = FAMILIES['all']
# ---- P rows with NEEDED columns (original dtypes) ----
tb = pq.read_table(f'{R}/training.parquet', columns=NEEDED + ['PHASE_mvt', 'TAXITIME_SEC_mvt'],
                   read_dictionary=[c for c in NEEDED if c.endswith('_mvt') and 'TIME' not in c] + ['APT', 'PHASE_mvt'])
tb = tb.filter(pc.equal(pc.cast(tb['PHASE_mvt'], pa.string()), 'DEP'))
y = tb['TAXITIME_SEC_mvt'].to_numpy().astype(float)
M = pc.cast(tb['MVT_TIME_UTC_mvt'], pa.int64()).to_numpy() / 1e6
S = pc.cast(tb['SCHED_TIME_UTC_mvt'], pa.int64()).to_numpy() / 1e6
miss = tb['AOBT_3_flt'].is_null().to_numpy(zero_copy_only=False)
os_all = M - S
inP = population(os_all, miss)
Pidx = np.where(inP)[0]
dP = tb.take(pa.array(Pidx)).to_pandas()
del tb
month_all = pd.to_datetime(M, unit='s').month.to_numpy()
X = build_base(dP.drop(columns=['PHASE_mvt', 'TAXITIME_SEC_mvt']))
X['log_off_sched'] = np.sign(X.off_sched) * np.log1p(X.off_sched.abs())
(X,) = align_categories(X)
cand = pd.read_parquet(f'{R}/exp62_cand_training.parquet')
pool = pd.read_parquet(f'{R}/exp62_pool_training.parquet')
assert np.array_equal(cand.index.to_numpy(), Pidx), 'P rows mismatch'
cand = cand.reset_index(drop=True)
yP = y[Pidx]; tP = (yP > TAIL_CAP).astype(int)
pool_art = (np.abs(y[pool['pos'].to_numpy()] - pool.off_sched.to_numpy()) < 60).astype(float)
XP = pd.concat([X[SPEC_FEATURES].reset_index(drop=True), cand[[c for c in EXTRA if not c.startswith('te_')]]], axis=1)
aptP = cand.apt.to_numpy(object); osP = cand.off_sched.to_numpy(float); missP = np.ones(len(cand), bool); monP = cand.month.to_numpy()
pb = np.load(f'{R}/exp62_pb.npy').astype(float); pbP = pb[Pidx]
print(f'setup {time.time()-t0:.0f}s, peak {peak():.2f} GB, P {len(Pidx)}', flush=True)

# ---- callsign-level stamping encodings (nested in-fold), from my compact cache ----
dd = load_dep(['APT', 'FLIGHT_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt'])
kc = K(S_(dd.APT), S_(dd.FLIGHT_mvt)); B = dd.BLOCK_TIME_UTC_mvt.to_numpy(); A3 = dd.AOBT_3_flt.to_numpy(); del dd
codes, uniq = pd.factorize(kc)
def per_key_month(mask, lab):
    Sm = np.zeros((len(uniq), 13)); Nm = np.zeros((len(uniq), 13))
    np.add.at(Sm, (codes[mask], month_all[mask]), lab[mask]); np.add.at(Nm, (codes[mask], month_all[mask]), 1)
    return Sm, Nm
A_mask = (~miss) & (A3 - S > 600) & (M - A3 > 0)
SA, NA_ = per_key_month(A_mask, (np.abs(B - S) < 60).astype(float))
rd_mask = miss & (os_all > 1800)
SR, NR = per_key_month(rd_mask, (np.abs(y - os_all) < 60).astype(float))
cP = codes[Pidx]
def te_nested(Sm, Nm, m, prior, w=4.0):
    """validation month m: exclude m; training rows (month k): exclude k and m."""
    s = Sm[cP].sum(1) - Sm[cP, m]; n = Nm[cP].sum(1) - Nm[cP, m]
    k = monP
    s_tr = s - np.where(k != m, Sm[cP, k], 0); n_tr = n - np.where(k != m, Nm[cP, k], 0)
    s = np.where(k == m, s, s_tr); n = np.where(k == m, n, n_tr)
    return (s + w * prior) / (n + w)
priorA = float((np.abs(B - S) < 60)[A_mask].mean()); priorR = float(pool_art.mean())

class WSpec(TailSpecialistX):
    """TailSpecialistX whose GBM gets sample weights and optionally the continuous mixing target."""
    def __init__(self, wmode='none', cont=False, **kw):
        super().__init__(**kw); self.wmode, self.cont = wmode, cont
    def fit(self, X, y, apt, off_sched, aobt_missing, pbv=None):
        TailSpecialist.fit(self, X, y, apt, off_sched, aobt_missing)     # table + iso (+ base gbm, replaced below)
        P = population(off_sched, aobt_missing)
        os_ = np.asarray(off_sched, float)
        gap = np.maximum(os_ - pbv, 1.0)
        w = {'none': np.ones_like(gap), 'abs': gap, 'sq': gap ** 2}[self.wmode]
        w = w / w[P].mean()
        if self.cont:
            lab = np.clip((np.asarray(y, float) - pbv) / gap, 0, 1); params = dict(SPEC_PARAMS, objective='cross_entropy')
        else:
            lab = (np.asarray(y, float) > TAIL_CAP).astype(float); params = SPEC_PARAMS
        feats = SPEC_FEATURES + self.extra
        cat = [c for c in feats if str(X[c].dtype) == 'category']
        mono = [1 if c in ('off_sched', 'log_off_sched') else 0 for c in feats]
        self.gbms_ = [lgb.train(dict(params, seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2, num_threads=2,
                                     monotone_constraints=mono),
                                lgb.Dataset(X.loc[P, feats], lab[P], weight=w[P], categorical_feature=cat),
                                num_boost_round=self.rounds) for s in self.seeds]
        return self
    def predict(self, X, apt, off_sched, aobt_missing):
        P = population(off_sched, aobt_missing); out = np.full(len(X), np.nan)
        Xp = X.loc[P, SPEC_FEATURES + self.extra]; os_v, apt_v = np.asarray(off_sched, float)[P], np.asarray(apt)[P]
        gbm = np.mean([m.predict(Xp) for m in self.gbms_], axis=0)
        out[P] = np.clip((gbm + self._table_predict(apt_v, os_v) + self._iso_predict(apt_v, os_v)) / 3.0, 0, 1)
        return out, P

ARMS = {  # name: (extra features, wmode, continuous label, add callsign TE)
    'ctl_x':        (EXTRA, 'none', False, False),   # should ~= exp62 spec_b (v7)
    'x_w_abs':      (EXTRA, 'abs', False, False),
    'x_w_sq':       (EXTRA, 'sq', False, False),
    'x_cont_sq':    (EXTRA, 'sq', True, False),
    'x_cs':         (EXTRA, 'none', False, True),
    'x_cs_w_abs':   (EXTRA, 'abs', False, True),
    'base_w_abs':   ([], 'abs', False, False),        # v6 specialist inputs, weighted
}
out = {k: np.full(len(cand), np.nan) for k in ARMS}
for m in range(1, 13):
    va, tr = monP == m, monP != m
    Xf = XP.copy()
    pk = (pool.month != m).to_numpy()
    te_tr, te_va = te_features(cand[tr], tP[tr], pool[pk], pool_art[pk], cand[va])
    for c in te_tr.columns:
        col = np.full(len(cand), np.nan); col[tr], col[va] = te_tr[c].to_numpy(), te_va[c].to_numpy(); Xf[c] = col
    Xf['te_cs_A_stamp'] = te_nested(SA, NA_, m, priorA)
    Xf['te_cs_rd_art'] = te_nested(SR, NR, m, priorR)
    for name, (ex, wm, cont, cs) in ARMS.items():
        ex2 = list(ex) + (['te_cs_A_stamp', 'te_cs_rd_art'] if cs else [])
        if not ex2:
            ex2 = []
        sp = WSpec(wmode=wm, cont=cont, extra=ex2, threads=2)
        sp.fit(Xf[tr], yP[tr], aptP[tr], osP[tr], missP[tr], pbv=pbP[tr])
        out[name][va] = sp.predict(Xf[va], aptP[va], osP[va], missP[va])[0]
    print(f'fold {m:02d} done {time.time()-t0:.0f}s peak {peak():.2f} GB', flush=True)
np.save(f'{D}/diag_e_specP.npy', np.vstack([out[k] for k in ARMS]))
# ---- evaluate through the full mixture ----
v6, v7, bl = preds()
spec = np.load(f'{R}/exp62_spec_P.npy')
print('ctl_x vs exp62 spec_b: max|dp| %.4f, corr %.5f' % (np.abs(out['ctl_x'] - spec[1]).max(), np.corrcoef(out['ctl_x'], spec[1])[0, 1]))
sc = np.isin(month_all, [1, 7])
def mix(p):
    new = v7.copy(); new[Pidx] = np.clip(p * osP + (1 - p) * pbP, 0, None); return new
ref = {'v7(exp62 b)': v7, 'ctl_x(refit)': mix(out['ctl_x'])}
print(f"{'arm':14s} {'Brier':>8s} {'12m RMSE':>9s} | vs v7: 12m d [CI]            scored d [CI]           | vs ctl_x 12m d [CI]")
for name in ARMS:
    new = mix(out[name])
    b1 = boot(y, new, v7, 500); b2 = boot(y[sc], new[sc], v7[sc], 500); b3 = boot(y, new, ref['ctl_x(refit)'], 500)
    print(f'{name:14s} {np.mean((out[name]-tP)**2):8.5f} {rmse(y,new):9.3f} | {b1[0]:+6.2f} [{b1[2]:+6.2f},{b1[3]:+6.2f}]  {b2[0]:+6.2f} [{b2[2]:+6.2f},{b2[3]:+6.2f}] | {b3[0]:+6.2f} [{b3[2]:+6.2f},{b3[3]:+6.2f}]')
print(f'v7 Brier {np.mean((spec[1]-tP)**2):.5f}  v7 12m RMSE {rmse(y, v7):.3f}; total {time.time()-t0:.0f}s peak {peak():.2f} GB')
