"""m03c -- does MVT_ID order carry information about y beyond v9's prediction? (leak test)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from prc.config import CACHE
M = CACHE / 'major'
d = pd.read_parquet(M / 'dep_training.parquet', columns=['APT', 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'y', 'month'])
d['res'] = d.y - np.load(M / 'v9s_oof.npy')
d['day'] = d.MVT_TIME_UTC_mvt.dt.floor('D')
g = d.groupby(['APT', 'day'], observed=True)
d['r_id'] = g.MVT_ID_mvt.rank(); d['r_s'] = g.SCHED_TIME_UTC_mvt.rank(); d['r_m'] = g.MVT_TIME_UTC_mvt.rank()
d['n'] = g.y.transform('size')
d['x_s'] = (d.r_id - d.r_s) / d.n; d['x_m'] = (d.r_id - d.r_m) / d.n
k = (d.y <= 7200)
for a, h in d[k].groupby('APT', observed=True):
    print(a, 'spearman(res, idrank-schedrank) %.4f   (res, idrank-mvtrank) %.4f   (y, idrank-schedrank) %.4f' % (
        spearmanr(h.res, h.x_s)[0], spearmanr(h.res, h.x_m)[0], spearmanr(h.y, h.x_s)[0]))
