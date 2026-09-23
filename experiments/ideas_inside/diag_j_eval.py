from common import *
d = load_dep(['TAXITIME_SEC_mvt']); y = d.TAXITIME_SEC_mvt.to_numpy()
rows = []; allz = {k: [] for k in ['idx', 'direct', 'direct2', 'resid', 'log']}
for m in [3, 6, 9, 12, 1, 7]:
    z = np.load(f'{D}/diag_j_m{m:02d}.npz')
    for k in allz: allz[k].append(z[k])
Z = {k: np.concatenate(v) for k, v in allz.items()}
yy = y[Z['idx']]; bulk = yy <= 7200
def rep(sel, tag):
    base = Z['direct'][sel]; yv = yy[sel]; r0 = yv - base
    print(f'--- {tag}: n {sel.sum():,}  direct solo {rmse(yv, base):.3f}')
    for k in ['direct2', 'resid', 'log']:
        p = Z[k][sel]; rho = np.corrcoef(r0[bulk[sel]], (yv - p)[bulk[sel]])[0, 1]
        bl = 0.5 * base + 0.5 * p
        dd = p - base; w = np.clip(np.dot(yv - base, dd) / np.dot(dd, dd), 0, 1)
        b = boot(yv, bl, base, 200)
        print(f'  {k:8s} solo {rmse(yv, p):.3f}  rho(bulk resid) {rho:.4f}  50/50 blend d {b[0]:+.3f} [{b[2]:+.3f},{b[3]:+.3f}]  w* {w:.2f}')
m_all = np.concatenate([np.full(len(np.load(f"{D}/diag_j_m{m:02d}.npz")['idx']), m) for m in [3, 6, 9, 12, 1, 7]])
rep(np.isin(m_all, [3, 6, 9, 12]), 'screening months 3,6,9,12 (route A)')
rep(np.isin(m_all, [1, 7]), 'scored months 1,7 (route A)')
