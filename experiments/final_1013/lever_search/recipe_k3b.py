# K3b recipe (NOT applied to the repo). Insert in assemble.assemble() after q = join_complement(C, win, out):
K3B_EDGES = [4200, 5000, 6000, 8000]                       # off_lobt buckets (3606,4200] (4200,5000] (5000,6000] (6000,8000] (8000,inf)
K3B_DELTA = np.array([-0.58, -11.60, 97.55, 241.66, 1094.49])  # full-2025 mean of (y - v29 OOF) per bucket, A/B/C non-projected rows
m = C.ABC & np.isfinite(C.off_lobt) & (C.off_lobt > WINDOW) & (q == out)   # rows the window / join complement did not move
g = q.astype(float); g[m] += K3B_DELTA[np.digitize(C.off_lobt[m], K3B_EDGES)]
g = np.rint(np.clip(g, 0, HI))
w = np.isfinite(C.off_lobt); g[w] = np.minimum(np.maximum(g[w], np.ceil(np.maximum(0, C.off_lobt[w] - WINDOW))), np.floor(C.off_lobt[w] + WINDOW))
cap = np.isin(C.apt, CAP_APTS); g[cap] = np.minimum(g[cap], CAP)
assert ((g != q) <= m).all()      # only K3b rows move; expected on final: 11,574 rows, 36 by +1094
q = g
