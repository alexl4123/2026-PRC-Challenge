"""Inside P: does callsign-level stamping history (other months) add to the v7 specialist?"""
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d = load_dep(['APT', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
B = d.BLOCK_TIME_UTC_mvt.to_numpy(); A3 = d.AOBT_3_flt.to_numpy()
os_ = M - S; miss = np.isnan(A3); P = miss & (os_ > 7200)
month = pd.to_datetime(M, unit='s').month.to_numpy()
apt = S_(d.APT); cs = S_(d.FLIGHT_mvt)
v6, v7, bl = preds()
pb = np.load(f'{R}/exp62_pb.npy').astype(float)
spec = np.load(f'{R}/exp62_spec_P.npy')      # rows: v6 spec, v6x spec on P rows in DEP order
Pi = np.where(P)[0]
# reconstruct v6/v7 inside P: pred = p*os + (1-p)*pb
for k, ref in [(0, v6), (1, v7)]:
    rec = spec[k] * os_[Pi] + (1 - spec[k]) * pb[Pi]
    print(f'reconstruct spec row {k}: max |diff| inside P = {np.abs(rec - ref[Pi]).max():.4f}')
yP = y[Pi]; tail = (yP > 7200).astype(float); mP = month[Pi]; aP = apt[Pi]
# ---- sources for callsign encodings (label known only in training months) ----
kc = K(apt, cs)
srcs = {
  'P_tail': (P, (y > 7200).astype(float)),                                   # tail inside P
  'rd_art': (miss & (os_ > 1800), (np.abs(y - os_) < 60).astype(float)),    # artefact in route-D pool
  'A_stamp': ((~miss) & (A3 - S > 600) & (M - A3 > 0), (np.abs(B - S) < 60).astype(float)),  # delayed route-A stamping
}
# transductive, label-free: fraction of this callsign's departures with no NM record, own month excluded? (observable in ranking) -> use ALL months incl own (no label)
codes, uniq = pd.factorize(kc)
nD = np.bincount(codes, weights=miss.astype(float)); nA = np.bincount(codes)
cs_rd_share = ((nD[codes] - miss) / np.maximum(nA[codes] - 1, 1))[Pi]     # leave-self-out, label free
cs_n = (nA[codes] - 1)[Pi]
feats = {'cs_rd_share(label-free)': cs_rd_share}
W = 4.0
for nm, (mask, lab) in srcs.items():
    Sm = np.zeros((len(uniq), 13)); Nm = np.zeros((len(uniq), 13))
    np.add.at(Sm, (codes[mask], month[mask]), lab[mask]); np.add.at(Nm, (codes[mask], month[mask]), 1)
    c = codes[Pi]
    s_ex = Sm[c].sum(1) - Sm[c, mP]; n_ex = Nm[c].sum(1) - Nm[c, mP]
    # prior = airport rate of this source (all months; small leak of airport mean, negligible)
    ar = pd.Series(lab[mask]).groupby(apt[mask]).mean()
    prior = pd.Series(aP).map(ar).fillna(lab[mask].mean()).to_numpy()
    feats[f'te_cs_{nm}'] = (s_ex + W * prior) / (n_ex + W)
    feats[f'n_cs_{nm}'] = n_ex
print(f'\nP rows {len(Pi)}, tail {int(tail.sum())}; LIRF {int((aP=="LIRF").sum())}')
for f, v in feats.items():
    if f.startswith('n_'):
        print(f'{f:28s} coverage(n>0) {np.mean(v > 0):.3f}  LIRF {np.mean(v[aP=="LIRF"] > 0):.3f}')
        continue
    ok = np.isfinite(v)
    lir = ok & (aP == 'LIRF')
    print(f'{f:28s} AUC {roc_auc_score(tail[ok], v[ok]):.3f}  LIRF AUC {roc_auc_score(tail[lir], v[lir]):.3f}')
for k, nm in [(0, 'v6 spec'), (1, 'v7 spec')]:
    lir = aP == 'LIRF'
    print(f'{nm:28s} AUC {roc_auc_score(tail, spec[k]):.3f}  LIRF AUC {roc_auc_score(tail[lir], spec[k][lir]):.3f}')
# ---- LOMO logistic stack on top of v7 spec ----
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
base_p = spec[1]
def stack(cols):
    X = np.column_stack([logit(base_p)] + [c for c in cols])
    out = np.empty(len(Pi))
    for m in range(1, 13):
        tr, va = mP != m, mP == m
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], tail[tr])
        out[va] = lr.predict_proba(X[va])[:, 1]
    return out
arms = {
  'recal only (logit v7)': [],
  '+te_cs_P_tail': [logit(feats['te_cs_P_tail']), (feats['n_cs_P_tail'] > 0).astype(float)],
  '+te_cs_rd_art': [logit(feats['te_cs_rd_art']), (feats['n_cs_rd_art'] > 0).astype(float)],
  '+te_cs_A_stamp': [logit(feats['te_cs_A_stamp'])],
  '+cs_rd_share': [cs_rd_share],
  '+all4': [logit(feats['te_cs_P_tail']), (feats['n_cs_P_tail'] > 0).astype(float), logit(feats['te_cs_rd_art']),
            (feats['n_cs_rd_art'] > 0).astype(float), logit(feats['te_cs_A_stamp']), cs_rd_share],
}
sc = np.isin(month, [1, 7])
for nm, cols in arms.items():
    p = stack(cols)
    new = v7.copy(); new[Pi] = p * os_[Pi] + (1 - p) * pb[Pi]
    bb = boot(y, new, v7, n=500)
    bs = boot(y[sc], new[sc], v7[sc], n=500)
    lir = aP == 'LIRF'
    print(f'{nm:24s} Brier {np.mean((p-tail)**2):.5f} (v7 {np.mean((base_p-tail)**2):.5f}) LIRF AUC {roc_auc_score(tail[lir], p[lir]):.3f} | '
          f'12m d {bb[0]:+.3f} [{bb[2]:+.3f},{bb[3]:+.3f}]  scored d {bs[0]:+.3f} [{bs[2]:+.3f},{bs[3]:+.3f}]')
print('peak GB', peak())
