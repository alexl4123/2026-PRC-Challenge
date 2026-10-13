"""Candidate stand rules: exactness on 2025 + OOF effect, and the 2026 label-free effect on v28."""
import numpy as np, pandas as pd
from stand_common import load_2025, load_2026, evaluate, apply_2026

c = load_2025(); k = load_2026()
NA = np.nan


def rules(d):
    Tam = (d['T'] - d.a_m).to_numpy(); Tlk = (d['T'] - d.a_lk).to_numpy()
    out = {}
    for tau in (3600, 7200, 10800, 14400, 21600):
        out[f'U1 any a_m, T-a_m>{tau}'] = (np.full(len(d), NA), np.where(Tam > tau, Tam, NA))
        out[f'U2 link n_after=0, T-a_lk>{tau}'] = (np.full(len(d), NA), np.where((d.n_after_lk == 0) & (Tlk > tau), Tlk, NA))
    for tau in (300, 600, 900):
        out[f'L1 any a_m, T-a_m<={tau}'] = (np.where(Tam <= tau, Tam, NA), np.full(len(d), NA))
        out[f'L2 a_m not ours & next dep = a_m rotation, T-a_m<={tau}'] = (np.where((Tam <= tau) & ~d.m_link & d.nxt_link, Tam, NA), np.full(len(d), NA))
    return out


R25, R26 = rules(c), rules(k)
rows = []
for nm in R25:
    r = evaluate(c, *R25[nm], nm)
    apply_2026(k, *R26[nm], nm)
    if r['exc']:
        e = r['exc_mask']; y = c.y.to_numpy(); lo, hi = R25[nm]
        over = np.where(np.isfinite(hi), y - hi, lo - y)[e]
        print(f'    exceptions: APT {c.APT[e].value_counts().head(5).to_dict()} | exceedance s q0/50/100 {np.percentile(over, [0, 50, 100]).round(0)} | P {int(c.P[e].sum())} D {int(c.D[e].sum())}')
