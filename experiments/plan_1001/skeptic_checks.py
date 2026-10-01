"""Skeptic checks (read-only, no training).
A. Arm O expected blend gain: single-seed adsb2 members vs 3-seed members, seq pool (27 fits) fixed, NNLS refit per route
   on screening months (v24 structure). 3 -> 6 seeds ~ (1 -> 3 gain) / 4 under MSE(n) = MSE_inf + s2/n.
B. Robustness of U (sd-tercile weights), U+T, and (d) airport-shrunk weights:
   B1 role swap: fit on the 8 non-screening months, judge on the 4 screening months (baseline also refit there).
   B2 leave-one-month-out fit (11 months) for baseline AND variant, judged on every month.
   B3 overlap: U + (d) combined vs each alone (standard fit on screening months).
"""
import sys
ROOT = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026'
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.optimize import nnls
from prc.config import CACHE
from prc.validation import block_bootstrap

FAMS = ['mix', 'nn', 'perapt', 'lintree', 'routed', 'xgboost', 'w3']
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
SCREEN = [3, 6, 9, 12]
V = np.load(CACHE / 'w2' / 'v24s_oof.npz', allow_pickle=True)
y, month, P, D, f24, corr = V['y'], V['month'], V['P'], V['D'], V['f'], V['corr']
N = len(y)


def oof(folder, fam, seeds, suffix='.npy'):
    o = np.zeros(N)
    for s in seeds:
        for m in range(1, 13):
            o[month == m] += np.load(folder / f'{fam}__seed{s:02d}__m{m:02d}{suffix}') / len(seeds)
    return o


rw = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['apt', 'date_loc'])
block = pd.factorize(rw.apt.astype(str) + '_' + rw.date_loc.astype(str))[0]
apt = rw.apt.astype(str).to_numpy()
A, DnP = ~D & ~P, D & ~P
SCR = np.isin(month, SCREEN)
held, S = ~SCR, np.isin(month, [1, 7])
r = lambda f, k: float(np.sqrt(np.mean((y[k] - f[k]) ** 2)))

seqpool = np.mean([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [0, 1, 2]) for v in POOL], axis=0)
Ms = {s: {f: oof(CACHE / 'cv' / 'adsb2', f, [s]) for f in FAMS} for s in (0, 1, 2)}
M3 = {f: np.mean([Ms[s][f] for s in (0, 1, 2)], axis=0) for f in FAMS}


def serve(Mem, fitmask):
    X = np.column_stack([Mem[f] for f in FAMS] + [seqpool])
    fo = f24.copy()
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        w, _ = nnls(X[fitmask & g & ~P], y[fitmask & g & ~P])
        fo[gD] = np.clip(X[gD] @ w + (0.5 * corr[gD] if isD else 0), 0, None)
    return fo


print('=== A. arm O extrapolation (seq pool fixed) ===')
f3 = serve(M3, SCR)
print(f'3-seed members: held {r(f3, held):.3f} (v24 {r(f24, held):.3f}) scored {r(f3, S):.3f}')
mse = lambda f, k: float(np.mean((y[k] - f[k]) ** 2))
g_h, g_s = [], []
for s in (0, 1, 2):
    f1 = serve(Ms[s], SCR)
    g_h.append(r(f1, held) - r(f3, held)); g_s.append(r(f1, S) - r(f3, S))
    print(f'  seed {s} single-seed members: held {r(f1, held):.3f} (+{g_h[-1]:.3f} vs 3-seed)  scored {r(f1, S):.3f} (+{g_s[-1]:.3f})')
print(f'1 -> 3 seed blend gain: held {np.mean(g_h):.3f}  scored {np.mean(g_s):.3f}; 3 -> 6 extrapolated ~ /4: held '
      f'{np.mean(g_h) / 4:.3f}  scored {np.mean(g_s) / 4:.3f}')
del Ms

# ---------- B: U, T, (d) ----------
B = np.array([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], '.bulk.npy') for v in POOL for s in (0, 1, 2)])
sd = B.std(0); del B
dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2]) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2])
X = np.column_stack([M3[f] for f in FAMS] + [seqpool])


def build(fitmask, applymask, use_U=True, use_T=False, use_apt=False):
    """returns predictions on applymask rows (outside P) for the structure, others = f24"""
    out = f24.copy()
    q = np.quantile(sd[fitmask & ~P], [1 / 3, 2 / 3])
    b = np.digitize(sd, q) if use_U else np.zeros(N, int)
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        w0, _ = nnls(X[fitmask & g & ~P], y[fitmask & g & ~P])
        for c in np.unique(b):
            gg = g & (b == c)
            k = fitmask & gg & ~P
            w, _ = nnls(X[k], y[k])
            if use_apt and not isD:
                for a_ in np.unique(apt):
                    ka = k & (apt == a_)
                    wa, _ = nnls(X[ka], y[ka])
                    sel = gD & (b == c) & (apt == a_) & applymask
                    out[sel] = np.clip(X[sel] @ (0.5 * w + 0.5 * wa), 0, None)
                continue
            base = X @ w
            lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k])) if use_T else 0.0
            sel = gD & (b == c) & applymask
            out[sel] = np.clip(base[sel] + lam * dl[sel] + (0.5 * corr[sel] if isD else 0), 0, None)
    return out


def rep(name, fn, fo, k_eval_list):
    out = []
    for nm, k in k_eval_list:
        bb = block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=500)
        out.append(f'{nm} {bb["point"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(N, bool); keep[np.argsort(gain)[::-1][:100]] = False
    k0 = k_eval_list[0][1]
    print(f'{name:38s} ' + '  '.join(out) + f'  w/o top100 {r(fn, k0 & keep) - r(fo, k0 & keep):+.3f}'
          + f'  top100 share of gain {gain[~keep & k0].sum() / max(gain[k0].sum(), 1e-9):.2f}')


allm = np.ones(N, bool)
print('\n=== B0. standard (fit screening months, judge held-out) vs v24-structure refit ===')
fo = build(SCR, allm, use_U=False)
print(f'baseline refit held {r(fo, held):.3f} vs v24 {r(f24, held):.3f}')
EV = [('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)]
fU = build(SCR, allm, True, False); rep('U', fU, fo, EV)
fUT = build(SCR, allm, True, True); rep('U+T', fUT, fo, EV)
fT = build(SCR, allm, False, True); rep('T', fT, fo, EV)
fd = build(SCR, allm, False, False, True); rep('(d) airport shrunk', fd, fo, EV)
fUd = build(SCR, allm, True, False, True); rep('U x (d) combined', fUd, fo, EV)

print('\n=== B1. role swap: fit on the 8 non-screening months, judge on the 4 screening months ===')
EV1 = [('screen4', SCR)]
fo1 = build(held, allm, use_U=False)
rep('U (swap)', build(held, allm, True, False), fo1, EV1)
rep('U+T (swap)', build(held, allm, True, True), fo1, EV1)
rep('T (swap)', build(held, allm, False, True), fo1, EV1)
rep('(d) (swap)', build(held, allm, False, False, True), fo1, EV1)

print('\n=== B2. leave-one-month-out fits (baseline and variant both LOMO), judged per scope ===')
fo2, fU2, fUT2, fd2 = f24.copy(), f24.copy(), f24.copy(), f24.copy()
for m in range(1, 13):
    tr, te = month != m, month == m
    fo2[te] = build(tr, te, False)[te]
    fU2[te] = build(tr, te, True, False)[te]
    fUT2[te] = build(tr, te, True, True)[te]
    fd2[te] = build(tr, te, False, False, True)[te]
EV2 = [('12m', allm), ('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)]
print(f'LOMO baseline 12m {r(fo2, allm):.3f} held {r(fo2, held):.3f} (v24 {r(f24, allm):.3f} / {r(f24, held):.3f})')
rep('U (LOMO)', fU2, fo2, EV2)
rep('U+T (LOMO)', fUT2, fo2, EV2)
rep('(d) (LOMO)', fd2, fo2, EV2)
print('months better U+T LOMO:', sum(r(fUT2, month == m) < r(fo2, month == m) for m in range(1, 13)), '/12;  U LOMO:',
      sum(r(fU2, month == m) < r(fo2, month == m) for m in range(1, 13)), '/12')

print('\n=== B4. Jan <-> Jul single-month cross-fit (GREKI scheme) ===')
for tr_m, te_m in ((1, 7), (7, 1)):
    tr, te = month == tr_m, month == te_m
    b0 = build(tr, te, False)
    for nm, kw in (('U', (True, False)), ('U+T', (True, True)), ('T', (False, True))):
        fn = build(tr, te, *kw)
        print(f'  fit {tr_m} judge {te_m}: {nm:4s} {r(fn, te) - r(b0, te):+.3f}')
