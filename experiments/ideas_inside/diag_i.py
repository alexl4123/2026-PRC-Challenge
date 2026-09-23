"""Ensembling tests on existing OOF members (outside P, where both are bulk models)."""
from common import *
import lightgbm as lgb
d = load_dep(['APT', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
month = pd.to_datetime(M, unit='s').month.to_numpy(); hour = pd.to_datetime(M, unit='s').hour.to_numpy()
apt = S_(d.APT); del d
P = np.isnan(A3) & (M - S > 7200)
v6, v7, bl = preds()
L, N = bl[0].astype(float), bl[1].astype(float)
o = ~P; sc = np.isin(month, [1, 7])
yo, Lo, No, ao, mo = y[o], L[o], N[o], apt[o], month[o]
def wopt(yy, a, b):
    # argmin_w ||y - ((1-w)a + w b)||^2
    dd = b - a; return float(np.clip(np.dot(yy - a, dd) / np.dot(dd, dd), 0, 1))
base = 0.5 * Lo + 0.5 * No
print(f'outside P: LGB {rmse(yo, Lo):.3f} NET {rmse(yo, No):.3f} 50/50 {rmse(yo, base):.3f}; residual corr (bulk) {np.corrcoef((yo-Lo)[yo<=7200], (yo-No)[yo<=7200])[0,1]:.4f}')
print('per-month optimal w on the net (in-month, descriptive):')
bulk = yo <= 7200
rows = []
for m in range(1, 13):
    k = (mo == m) & bulk
    rows.append((m, wopt(yo[k], Lo[k], No[k]), rmse(yo[k], Lo[k]), rmse(yo[k], No[k]), rmse(yo[k], 0.5*Lo[k]+0.5*No[k])))
print(pd.DataFrame(rows, columns=['m', 'w*', 'lgb_bulk', 'net_bulk', 'blend_bulk']).round(3).to_string(index=False))
print('per-airport optimal w (all months, bulk rows):', {a: round(wopt(yo[(ao == a) & bulk], Lo[(ao == a) & bulk], No[(ao == a) & bulk]), 3) for a in np.unique(ao)})
# nested LOMO: global w, per-airport w, per-airport x AOBT-missing w (weights fitted on bulk rows of other months)
miss_o = np.isnan(A3[o])
arms = {}
for nm, keyf in [('global', lambda: np.zeros(len(yo), int)), ('apt', lambda: pd.factorize(ao)[0]),
                 ('apt x routeD', lambda: pd.factorize(K(ao, miss_o.astype(str).astype(object)))[0]),
                 ('apt x hourband', lambda: pd.factorize(K(ao, (hour[o] // 6).astype(str).astype(object)))[0])]:
    g = keyf(); pred = np.empty(len(yo))
    for m in range(1, 13):
        tr, va = (mo != m) & bulk, mo == m
        for kk in np.unique(g[va]):
            t = tr & (g == kk)
            w = wopt(yo[t], Lo[t], No[t]) if t.sum() > 1000 else 0.5
            s = va & (g == kk); pred[s] = (1 - w) * Lo[s] + w * No[s]
    arms[nm] = pred
# stacking meta-model (nested LOMO), small LGB on [L, N, apt, route D, off_aobt3, hour]
Xs = pd.DataFrame(dict(L=Lo, N=No, d=No - Lo, apt=pd.Categorical(ao).codes, rd=miss_o.astype(int), oa=(M - A3)[o], hour=hour[o]))
pred = np.empty(len(yo)); rng = np.random.default_rng(0)
prm = dict(objective='regression', learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, verbose=-1, num_threads=2, seed=0, deterministic=True, force_row_wise=True)
for m in range(1, 13):
    tri = np.where((mo != m) & bulk)[0]; tri = np.sort(rng.choice(tri, 400_000, replace=False)); vai = np.where(mo == m)[0]
    mdl = lgb.train(prm, lgb.Dataset(Xs.iloc[tri], yo[tri] - 0.5 * (Lo[tri] + No[tri]), categorical_feature=['apt']), 150)
    pred[vai] = 0.5 * (Lo[vai] + No[vai]) + mdl.predict(Xs.iloc[vai])
arms['stack LGB meta'] = pred
sco = sc[o]
for nm, p in arms.items():
    b1 = boot(yo, p, base, 300); b2 = boot(yo[sco], p[sco], base[sco], 300)
    print(f'{nm:16s} outsideP 12m d {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored d {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
# fixed global weights, for the "net under-weighted?" question
for w in [0.6, 0.7, 0.8]:
    p = (1 - w) * Lo + w * No
    b1 = boot(yo, p, base, 300); b2 = boot(yo[sco], p[sco], base[sco], 300)
    print(f'fixed w={w}      outsideP 12m d {b1[0]:+.3f} [{b1[2]:+.3f},{b1[3]:+.3f}]  scored d {b2[0]:+.3f} [{b2[2]:+.3f},{b2[3]:+.3f}]')
print('peak', peak())
