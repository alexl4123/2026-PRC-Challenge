"""Skeptic check 3 (read-only): Jan<->Jul single-month cross-fit for (d), U+T, U+T+(d); and (d)'s gain by own-ground coverage at EGLL/LIRF."""
src = open('skeptic_checks2.py').read().split("allm = np.ones(N, bool)")[0]
exec(src)
for tr_m, te_m in ((1, 7), (7, 1)):
    tr, te = month == tr_m, month == te_m
    b0 = build(tr, te, False)
    for nm, kw in (('(d)', (False, False, True)), ('U+T', (True, True, False)), ('U+T+(d)', (True, True, True)), ('U', (True, False, False))):
        fn = build(tr, te, *kw)
        print(f'  fit {tr_m} judge {te_m}: {nm:8s} {r(fn, te) - r(b0, te):+.3f}   ' +
              ' '.join(f'{a_} {r(fn, te & (apt == a_)) - r(b0, te & (apt == a_)):+.2f}' for a_ in sorted(set(apt))))
gr = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_training.parquet', columns=['adsb_ground']).adsb_ground.to_numpy().astype(bool)
fo2, fd2 = f24.copy(), f24.copy()
for m in range(1, 13):
    tr, te = month != m, month == m
    fo2[te] = build(tr, te, False)[te]; fd2[te] = build(tr, te, False, False, True)[te]
for a_ in ('EGLL', 'LIRF', 'LEMD', 'LFPG'):
    for gnm, gk in (('ground', gr), ('no ground', ~gr)):
        k = (apt == a_) & gk
        print(f'(d) LOMO {a_} {gnm:9s} n {k.sum():7d}  dRMSE {r(fd2, k) - r(fo2, k):+.2f}')
