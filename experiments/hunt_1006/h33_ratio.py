import numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'; H = R + 'cache/hunt/'
src = open(H + 'h25_candidate.py').read().split('# ---- 2025 OOF')[0]; ns = {}; exec(src, ns); rules = ns['rules']
V = np.load(R + 'cache/w2/v28s_oof.npz'); f = V['f']; y = V['y']; month = V['month']
g, e, info, d = rules('tr', f)
for nm, h in (('A+B+C', g), ('A+B+C+E1', e)):
    ch = h != f; act = ((f - y) ** 2 - (h - y) ** 2)[ch].sum(); bnd = ((h - f) ** 2)[ch].sum()
    S = np.isin(month, [1, 7]) & ch
    print(f'{nm}: actual SSE gain / sum of squared moves = {act / bnd:.2f} (12m), {((f - y) ** 2 - (h - y) ** 2)[S].sum() / ((h - f) ** 2)[S].sum():.2f} (Jan+Jul)')
c = pd.read_csv(H + 'cand_moves.csv')
for col, rt in (('ABC', None), ('E1', None)):
    s2 = ((c[col] - c.v28) ** 2).sum()
    for ratio in (1.0, 1.5, 2.0):
        print(f' 2026 {col}: sum sq moves {s2:.4g}; if gain = {ratio} x moves^2 -> RMSE {np.sqrt(235.3996 ** 2 - ratio * s2 / 344841) - 235.3996:+.3f}')
