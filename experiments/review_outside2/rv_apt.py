"""Per-airport LightGBM as a blend member: does its decorrelation survive more training data?
Base72 features (the cluster perapt arm's inputs), clean bulk. usage: rv_apt.py <month> <ntrain> <rounds>"""
import sys, time
from common import *
import lightgbm as lgb
m, ntr, RN = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
X = pd.read_parquet(f'{R}/altfam_X.parquet')
y = np.load(f'{R}/altfam_y.npy'); mon = np.load(f'{R}/altfam_month.npy')
P = (X.aobt_missing.values == 1) & (X.off_sched.values > 7200); clean = (y <= 7200) & ~P
rng = np.random.default_rng(200 + m)
tr = np.sort(rng.choice(np.where(clean & (mon != m))[0], ntr, replace=False)); te = np.where(clean & (mon == m))[0]
X = X.iloc[np.concatenate([tr, te])].reset_index(drop=True); n0 = len(tr); del P
cats = ['APT', 'RUNWAY_mvt', 'PIER', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt']
PAR = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, deterministic=True, force_row_wise=True)
Xtr, Xte = X.iloc[:n0], X.iloc[n0:]; ytr, yt = y[tr], y[te]
pr = {}
pr['g0'] = lgb.train(dict(PAR, seed=0), lgb.Dataset(Xtr, ytr, categorical_feature=cats), RN).predict(Xte)
pr['g1'] = lgb.train(dict(PAR, seed=7, bagging_seed=8, feature_fraction_seed=9), lgb.Dataset(Xtr, ytr, categorical_feature=cats), RN).predict(Xte)
print(f'globals {time.time()-t0:.0f}s', flush=True)
at, ae = Xtr.APT.astype(str).to_numpy(), Xte.APT.astype(str).to_numpy(); pa = np.zeros(len(te))
for a in np.unique(at):
    i, j = at == a, ae == a
    b = lgb.train(dict(PAR, seed=0, num_leaves=127), lgb.Dataset(Xtr[i].drop(columns=['APT']), ytr[i], categorical_feature=[c for c in cats if c != 'APT']), RN)
    pa[j] = b.predict(Xte[j].drop(columns=['APT']))
pr['apt'] = pa
rm = lambda p: np.sqrt(np.mean((yt - p) ** 2)); rc = lambda a, b: np.corrcoef(yt - a, yt - b)[0, 1]
g2 = 0.5 * (pr['g0'] + pr['g1'])
print(f'm{m:02d} n {ntr} R {RN}: solo g0 {rm(pr["g0"]):.2f} g1 {rm(pr["g1"]):.2f} apt {rm(pr["apt"]):.2f} | rho(g0,g1) {rc(pr["g0"],pr["g1"]):.4f} rho(g0,apt) {rc(pr["g0"],pr["apt"]):.4f}')
print(f'   50/50 vs g0: g0+g1 {rm(g2)-rm(pr["g0"]):+.2f} | g0+apt {rm(0.5*pr["g0"]+0.5*pr["apt"])-rm(pr["g0"]):+.2f} | 2-seed global + apt (50/50) vs 2-seed global {rm(0.5*g2+0.5*pr["apt"])-rm(g2):+.2f} | 1/3 each vs 2-seed {rm((pr["g0"]+pr["g1"]+pr["apt"])/3)-rm(g2):+.2f}')
print('peak', peak(), f'{time.time()-t0:.0f}s')
