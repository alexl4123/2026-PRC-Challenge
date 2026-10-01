"""Skeptic check 2 (read-only): U+T+(d) bundle, standard and LOMO; per-airport and per-month breakdown of U+T LOMO."""
import sys
exec(open('skeptic_checks.py').read().split("print('=== A.")[0])
B = np.array([oof(CACHE / 'cv' / f'seqsw_{v}', 'seq', [s], '.bulk.npy') for v in POOL for s in (0, 1, 2)])
sd = B.std(0); del B
dl = oof(CACHE / 'cv' / 'seq', 'seq', [0, 1, 2]) - oof(CACHE / 'cv' / 'seq', 'seqnc', [0, 1, 2])
M3 = {f: oof(CACHE / 'cv' / 'adsb2', f, [0, 1, 2]) for f in FAMS}
X = np.column_stack([M3[f] for f in FAMS] + [seqpool])
def build(fitmask, applymask, use_U=True, use_T=False, use_apt=False):
    out = f24.copy()
    q = np.quantile(sd[fitmask & ~P], [1 / 3, 2 / 3])
    b = np.digitize(sd, q) if use_U else np.zeros(N, int)
    for g, gD, isD in ((~D, A, False), (D, DnP, True)):
        for c in np.unique(b):
            k = fitmask & g & (b == c) & ~P
            w, _ = nnls(X[k], y[k])
            base = X @ w
            if use_apt and not isD:
                for a_ in np.unique(apt):
                    ka = k & (apt == a_)
                    wa, _ = nnls(X[ka], y[ka])
                    sa = (apt == a_)
                    base[sa] = X[sa] @ (0.5 * w + 0.5 * wa)
            lam = float(np.dot(dl[k], (y - base)[k]) / np.dot(dl[k], dl[k])) if use_T else 0.0
            sel = gD & (b == c) & applymask
            out[sel] = np.clip(base[sel] + lam * dl[sel] + (0.5 * corr[sel] if isD else 0), 0, None)
    return out
def rep(name, fn, fo, ev):
    out = []
    for nm, k in ev:
        bb = block_bootstrap(y[k], fn[k], fo[k], block[k], n_boot=500)
        out.append(f'{nm} {bb["point"]:+.3f} [{bb["lo"]:+.3f},{bb["hi"]:+.3f}]')
    gain = (y - fo) ** 2 - (y - fn) ** 2
    keep = np.ones(N, bool); keep[np.argsort(gain)[::-1][:100]] = False
    k0 = ev[0][1]
    print(f'{name:30s} ' + '  '.join(out) + f'  w/o top100 {r(fn, k0 & keep) - r(fo, k0 & keep):+.3f} / scored {r(fn, S & keep) - r(fo, S & keep):+.3f}')
allm = np.ones(N, bool)
EV = [('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)]
fo = build(SCR, allm, False)
rep('U+T+(d) standard', build(SCR, allm, True, True, True), fo, EV)
fo2, fUT2, fUTd2 = f24.copy(), f24.copy(), f24.copy()
for m in range(1, 13):
    tr, te = month != m, month == m
    fo2[te] = build(tr, te, False)[te]
    fUT2[te] = build(tr, te, True, True)[te]
    fUTd2[te] = build(tr, te, True, True, True)[te]
EV2 = [('12m', allm), ('held', held), ('scored', S), ('Jan', month == 1), ('Jul', month == 7)]
rep('U+T (LOMO)', fUT2, fo2, EV2)
rep('U+T+(d) (LOMO)', fUTd2, fo2, EV2)
for nm, fn in (('U+T LOMO', fUT2), ('U+T+(d) LOMO', fUTd2)):
    print(nm, 'per airport 12m:', ' '.join(f'{a_} {r(fn, apt == a_) - r(fo2, apt == a_):+.2f}' for a_ in sorted(set(apt))))
    print(nm, 'per month:', ' '.join(f'{m} {r(fn, month == m) - r(fo2, month == m):+.2f}' for m in range(1, 13)))
    # SSE share of the gain by airport (12m)
    g = (y - fo2) ** 2 - (y - fn) ** 2
    print(nm, 'gain SSE share by airport:', ' '.join(f'{a_} {g[apt == a_].sum() / g.sum():.2f}' for a_ in sorted(set(apt))))
    print(nm, 'route D outside P share of gain:', round(float(g[DnP].sum() / g.sum()), 2), ' A/B/C share', round(float(g[A].sum() / g.sum()), 2))
