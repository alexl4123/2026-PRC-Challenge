"""Marginal value of a per-airport member INSIDE v8~ (full-data LGB + 3-seed net), vs a global member of the same size (control)."""
import sys
from common import *
m, ntr = int(sys.argv[1]), int(sys.argv[2])
z = np.load(f'{D}/rv_apt_m{m:02d}_{ntr}.npz'); te = z['te']
y = np.load(f'{R}/altfam_y.npy')[te]
v6, v7, bl = preds(); lgbf, net = v7[te], bl[1][te].astype(float); v8 = 0.5 * lgbf + 0.5 * net
rm = lambda p: np.sqrt(np.mean((y - p) ** 2)); rc = lambda a, b: np.corrcoef(y - a, y - b)[0, 1]
print(f'm{m:02d} clean bulk n={len(te)}: full LGB {rm(lgbf):.2f}  net {rm(net):.2f}  v8~ {rm(v8):.2f} | 1M global {rm(z["g0"]):.2f}  1M per-airport {rm(z["apt"]):.2f}')
print(f'  rho: LGBfull-net {rc(lgbf,net):.4f} | apt-LGBfull {rc(z["apt"],lgbf):.4f} apt-net {rc(z["apt"],net):.4f} apt-v8~ {rc(z["apt"],v8):.4f} | g1M-LGBfull {rc(z["g0"],lgbf):.4f} g1M-v8~ {rc(z["g0"],v8):.4f}')
for w in [0.2, 1/3, 0.5]:
    for nm in ['apt', 'g0']:
        p = (1 - w) * v8 + w * z[nm]
        b = boot(y, p, v8, 200)
        print(f'  w={w:.2f} + {nm:3s}: {rm(p)-rm(v8):+.2f} [{b[2]:+.2f},{b[3]:+.2f}]')
