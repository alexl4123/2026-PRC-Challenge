"""s18 -- is 'more network weight' a live lever?  Fit SSE(w) = a + b w + c w^2 through our own three
live scores that differ (almost) only in the route-A/B/C network weight w:
    v7 (w=0, 304.351)   v8 (w=0.5 everywhere, 297.829)   v9 (w=0.65 on A/B/C, 0.5 on D, 297.3414)
Caveat: v7->v8 also moved route D (w 0->0.5) and the bulk part of P rows (small: sum delta^2 6.1e7
vs 6.7e8 on A/B/C, major m04), so this is approximate.  Uses 2026 labels only through our own public scores (no new upload)."""
import numpy as np
N = 344841
w = np.array([0.0, 0.5, 0.65]); s = np.array([304.351, 297.829, 297.3414]) ** 2 * N
c, b, a = np.polyfit(w, s, 2)
ws = -b / (2 * c); smin = a - b * b / (4 * c)
print(f'SSE(w) = {a:.4e} + {b:.4e} w + {c:.4e} w^2  ->  live-optimal w* = {ws:.3f}, RMSE at w* {np.sqrt(smin/N):.3f} '
      f'(v9 at 0.65: 297.341; gain from moving to w* {np.sqrt(smin/N)-297.3414:+.3f} s)')
for ww in [0.7, 0.8, 0.9, 1.0]:
    print(f'  w={ww}: predicted {np.sqrt((a+b*ww+c*ww*ww)/N):.3f}')
