from common import *
rows = []
for which in ['training', 'ranking']:
    d = load_dep(['APT', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'EOBT_1_flt', 'LOBT_flt', 'IOBT_flt'], which)
    M = d.MVT_TIME_UTC_mvt.to_numpy(); mon = pd.to_datetime(M, unit='s').month
    f = pd.DataFrame(dict(apt=S_(d.APT), mon=mon, a3_m_lobt=d.AOBT_3_flt - d.LOBT_flt, a3_m_eobt=d.AOBT_3_flt - d.EOBT_1_flt,
                          off_iobt=M - d.IOBT_flt, iobt_m_sched=d.IOBT_flt - d.SCHED_TIME_UTC_mvt, iobt_present=d.IOBT_flt.notna()))
    f = f[np.isin(mon, [1, 7])]; f['set'] = '2025' if which == 'training' else '2026'
    rows.append(f)
    if which == 'ranking':
        tr = set(load_dep(['ADES_mvt'], 'training').ADES_mvt.astype(str)); ades = d.ADES_mvt.astype(str)
        print(f'ranking ADES seen in 2025: {ades.isin(tr).mean():.4f}')
f = pd.concat(rows)
g = f.groupby(['apt', 'set', 'mon']).agg(iobt_present=('iobt_present', 'mean'), a3_m_lobt_med=('a3_m_lobt', 'median'),
      a3_m_eobt_med=('a3_m_eobt', 'median'), iobt_m_sched_med=('iobt_m_sched', 'median'), off_iobt_med=('off_iobt', 'median'))
print(g.round(2).unstack(['set', 'mon']).to_string())
f2 = pd.read_parquet(f'{D}/stampfeat_training.parquet'); f3 = pd.read_parquet(f'{D}/stampfeat_ranking.parquet')
print('stamp feats medians training vs ranking:'); print(pd.concat([f2.median(), f3.median(), f2.notna().mean(), f3.notna().mean()], axis=1, keys=['med25', 'med26', 'cov25', 'cov26']).round(3).to_string())
