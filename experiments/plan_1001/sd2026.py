import sys
ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from prc.config import CACHE
POOL = ['ep30', 'ep40', 'ep50', 'ep60', 'ep80', 'ep40_d128', 'ep50_d128', 'ep40_l3', 'ep40_lr3']
xs = []
for v in POOL:
    fo = CACHE / f'seq_{v}_final'
    print(v, sorted(p.name for p in fo.glob('*'))[:8])
    for s in (0, 1, 2):
        xs.append(np.load(fo / f'seq__seed{s:02d}.npy').astype(float))
X = np.array(xs); sd = X.std(0)
rk = pd.read_parquet(CACHE / 'drivers_0926' / 'rows_ranking.parquet', columns=['apt', 'route'])
ok = (rk.route.astype(str) != 'P').to_numpy()
print('2026 finals pool sd (outside P): quantiles 1/3, 1/2, 2/3:', np.round(np.quantile(sd[ok], [1/3, .5, 2/3]), 1),
      ' share below 13.9:', round(float((sd[ok] < 13.9).mean()), 3), ' above 25.4:', round(float((sd[ok] > 25.4).mean()), 3))
print(pd.Series(sd[ok]).groupby(rk.apt.astype(str).to_numpy()[ok]).median().round(1).to_string())
