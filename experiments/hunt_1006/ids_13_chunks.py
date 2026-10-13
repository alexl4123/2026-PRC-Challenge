"""ids_13 -- the raw files are made of MVT-sorted chunks. What decides chunk membership? Does it relate to y beyond
the model (v28 OOF residual)? Training: all 12 monthly files in cache order (= raw order); ranking raw order."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from prc.io import load
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'FLIGHT_mvt']
def chunks(d):
    m = d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64')
    apt = d.APT.astype(str).values
    dm = np.diff(m)
    brk = np.r_[True, (apt[1:] != apt[:-1]) | (dm < -60)]   # noise is <= 12 s; any decrease > 60 s = new chunk
    d['cid'] = np.cumsum(brk)
    g = d.groupby('cid')
    d['csize'] = g.MVT_ID_mvt.transform('size')
    d['cspan'] = (g.MVT_TIME_UTC_mvt.transform('max') - g.MVT_TIME_UTC_mvt.transform('min')).dt.total_seconds() / 3600
    d['cpos'] = g.cumcount()
    return d
tr = chunks(load('training', columns=C))
rk = chunks(load('ranking', columns=C))
for name, d in [('training', tr), ('ranking', rk)]:
    print('=====', name, 'chunks', d.cid.max())
    c = d.groupby('cid').agg(n=('cid', 'size'), span=('cspan', 'first'), dep=('PHASE_mvt', lambda s: (s == 'DEP').mean()),
                             nm=('LOBT_flt', lambda s: s.notna().mean()), apt=('APT', 'first'))
    c['span_cat'] = pd.cut(c.span, [-1, 1, 6, 24, 1e9], labels=['<1h', '1-6h', '6-24h', '>24h'])
    print(c.groupby('span_cat')[['n', 'dep', 'nm']].agg(['count', 'mean']).round(3).to_string())
    print(' rows by span cat:', d.assign(sc=pd.cut(d.cspan, [-1, 1, 6, 24, 1e9], labels=['<1h', '1-6h', '6-24h', '>24h'])).groupby('sc').size().to_dict())
# DEP training: OOF residual by chunk span category
mt = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt', 'off_sched'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
mt['f'] = z['f']; mt['y'] = z['y']; mt['P'] = z['P']; mt['D'] = z['D']; mt['month'] = z['month']
dep = tr[tr.PHASE_mvt.eq('DEP')].merge(mt, on='MVT_ID_mvt')
dep['res'] = dep.y - dep.f
dep['sc'] = pd.cut(dep.cspan, [-1, 1, 6, 24, 1e9], labels=['<1h', '1-6h', '6-24h', '>24h'])
dep['szc'] = pd.cut(dep.csize, [0, 25, 70, 200, 1e9])
print('\nDEP training: residual by chunk span category')
print(dep.groupby('sc').agg(n=('res', 'size'), mean_res=('res', 'mean'), rmse=('res', lambda r: np.sqrt(np.mean(r ** 2))),
                            mean_y=('y', 'mean'), tail=('y', lambda s: (s > 7200).mean()), P=('P', 'mean'), D=('D', 'mean')).round(3).to_string())
print(dep.groupby('szc').agg(n=('res', 'size'), mean_res=('res', 'mean'), rmse=('res', lambda r: np.sqrt(np.mean(r ** 2))),
                             tail=('y', lambda s: (s > 7200).mean()), P=('P', 'mean')).round(3).to_string())
# within P: tail rate by chunk type
p = dep[dep.P]
print('\nP rows by chunk span cat: ', p.groupby('sc').agg(n=('y', 'size'), tail=('y', lambda s: (s > 7200).mean()), mean_res=('res', 'mean')).round(3).to_string())
from sklearn.metrics import roc_auc_score
for k in ['cspan', 'csize', 'cpos']:
    print(f'  AUC({k} -> tail) inside P: {roc_auc_score(p.y > 7200, p[k]):.3f}')
tr[['MVT_ID_mvt', 'cid', 'csize', 'cspan', 'cpos']].to_parquet(H + 'cache/hunt/ids_13_chunks_tr.parquet')
rk[['MVT_ID_mvt', 'cid', 'csize', 'cspan', 'cpos']].to_parquet(H + 'cache/hunt/ids_13_chunks_rk.parquet')
