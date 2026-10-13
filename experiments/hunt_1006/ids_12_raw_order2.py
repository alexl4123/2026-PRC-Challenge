"""ids_12 -- raw row order: chunks sorted by MVT? What are the small inversions? Is the sort key MVT (pre-noise),
or a key that differs for DEP (e.g. BLOCK)?"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
D = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
C = ['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
for f in ['training_2025-07-01_2025-08-01.parquet', 'ranking.parquet']:
    d = pd.read_parquet(D + f, columns=C)
    ph = d.PHASE_mvt.astype(str).values
    apt = np.where(ph == 'DEP', d.ADEP_mvt.astype(str), d.ADES_mvt.astype(str))
    m = d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64')
    b = d.BLOCK_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64')
    bnan = d.BLOCK_TIME_UTC_mvt.isna().values
    dm = np.diff(m)
    same = apt[1:] == apt[:-1]
    print('=====', f)
    neg = dm[same & (dm < 0)]
    print('  adjacent same-airport pairs', same.sum(), ' MVT decreases', len(neg), ' quantiles of decrease',
          np.quantile(-neg, [.1, .5, .9, .99, .999, 1]) if len(neg) else None)
    big = same & (dm < -3600)
    print('  decreases > 1h (chunk boundaries?)', big.sum())
    # chunks = maximal runs between big decreases or airport changes
    brk = np.r_[True, (~same) | (dm < -3600)]
    cid = np.cumsum(brk)
    cs = pd.Series(cid).value_counts()
    print('  chunks', cid.max(), ' size quantiles', cs.quantile([.1, .5, .9]).to_dict())
    # chunk time spans
    t = pd.DataFrame({'cid': cid, 'm': m, 'apt': apt})
    sp = t.groupby('cid').agg(apt=('apt', 'first'), m0=('m', 'min'), m1=('m', 'max'), n=('m', 'size'))
    sp['span_h'] = (sp.m1 - sp.m0) / 3600
    print('  chunk span hours quantiles', sp.span_h.quantile([.1, .5, .9, 1]).round(2).to_dict())
    print(sp.head(10).assign(m0=lambda x: pd.to_datetime(x.m0, unit='s'), m1=lambda x: pd.to_datetime(x.m1, unit='s')).to_string())
    # small decreases: which phases are involved?
    small = same & (dm < 0) & (dm >= -3600)
    i = np.flatnonzero(small)
    pp = pd.Series([ph[k] + '>' + ph[k + 1] for k in i]).value_counts()
    print('  small decreases by phase pair', pp.to_dict())
    # for DEP->DEP small decreases: compare BLOCK order (training only)
    if not bnan.all():
        dd = np.array([ph[k] == 'DEP' and ph[k + 1] == 'DEP' for k in i])
        k = i[dd]
        print('  DEP>DEP small MVT decreases:', len(k), ' of which BLOCK increases:', np.mean(b[k + 1] > b[k]))
        # all DEP->DEP adjacent pairs within chunk: P(BLOCK increasing) vs P(MVT increasing)
        dep2 = same & (ph[1:] == 'DEP') & (ph[:-1] == 'DEP') & (dm >= -3600)
        print('  DEP>DEP adjacent (in-chunk) pairs', dep2.sum(), ' P(MVT up)', np.mean(dm[dep2] >= 0), ' P(BLOCK up)', np.mean(np.diff(b)[dep2] >= 0))
        # magnitude of the small decreases for ARR>ARR, DEP>DEP
        for a_, b_ in [('ARR', 'ARR'), ('DEP', 'DEP'), ('ARR', 'DEP'), ('DEP', 'ARR')]:
            sel = np.array([ph[q] == a_ and ph[q + 1] == b_ for q in i])
            if sel.sum():
                print(f'   {a_}>{b_}: n={sel.sum()} decrease quantiles', np.quantile(-dm[i[sel]], [.5, .9, .99, 1]))
    else:
        for a_, b_ in [('ARR', 'ARR'), ('DEP', 'DEP'), ('ARR', 'DEP'), ('DEP', 'ARR')]:
            sel = np.array([ph[q] == a_ and ph[q + 1] == b_ for q in i])
            if sel.sum():
                print(f'   {a_}>{b_}: n={sel.sum()} decrease quantiles', np.quantile(-dm[i[sel]], [.5, .9, .99, 1]))
