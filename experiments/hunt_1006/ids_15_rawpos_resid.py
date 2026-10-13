"""ids_15 -- raw-file position vs v28 OOF residual (the decisive test for a row-order leak), per airport and inside P;
also the block structure of the raw files (sizes of runs of consecutive MVT ranks)."""
import sys
sys.path.insert(0, '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from prc.io import load
import warnings; warnings.filterwarnings('ignore')
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
pd.set_option('display.width', 250)
d = load('training', columns=['MVT_ID_mvt', 'APT', 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
d['pos'] = np.arange(len(d))
d['APT'] = d.APT.astype(str)
d['day'] = d.MVT_TIME_UTC_mvt.dt.floor('D')
d['m'] = d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64')
g = d.groupby(['APT', 'day'])
d['r_pos'] = g.pos.rank()
d['r_mvt'] = g.m.rank(method='first')
d['n'] = g.pos.transform('size')
d['dpos'] = (d.r_pos - d.r_mvt) / d.n
# block structure: in file order, runs where MVT rank increases by exactly 1 (within airport-day)
d = d.sort_values('pos')
same = (d.APT.values[1:] == d.APT.values[:-1]) & (d.day.values[1:] == d.day.values[:-1])
step = np.diff(d.r_mvt.values)
cont = same & (np.abs(step) <= 3)
brk = np.r_[True, ~cont]
bl = pd.Series(np.cumsum(brk)).value_counts()
print('blocks of consecutive MVT ranks: count', len(bl), ' size quantiles', bl.quantile([.05, .25, .5, .75, .95]).to_dict())
mt = pd.read_parquet(H + 'cache/major/dep_training.parquet', columns=['MVT_ID_mvt'])
z = np.load(H + 'cache/w2/v28s_oof.npz')
mt['f'] = z['f']; mt['y'] = z['y']; mt['P'] = z['P']; mt['D'] = z['D']
dep = d[d.PHASE_mvt.eq('DEP')].merge(mt, on='MVT_ID_mvt')
dep['res'] = dep.y - dep.f
rows = []
for a, x in dep.groupby('APT'):
    rows.append({'apt': a, 'rho(dpos,res)': spearmanr(x.dpos, x.res)[0], 'rho(|dpos|,|res|)': spearmanr(x.dpos.abs(), x.res.abs())[0]})
print(pd.DataFrame(rows).round(4).to_string())
p = dep[dep.P]
print('inside P: AUC(dpos -> tail)', round(roc_auc_score(p.y > 7200, p.dpos), 3), ' LIRF P:',
      round(roc_auc_score(p[p.APT == 'LIRF'].y > 7200, p[p.APT == 'LIRF'].dpos), 3),
      ' rho(dpos,res) in P', round(spearmanr(p.dpos, p.res)[0], 4))
