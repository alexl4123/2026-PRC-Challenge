"""Per-month coverage + distribution of the cached feature files, 2025 (1,2,6,7) vs final 2026."""
import sys, numpy as np, pandas as pd
R = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/'
O = '/tmp/claude-1000/-home-thinklex-Dropbox-2025-phd-2026-35-PRC-Data-Challenge-OpenSky-00-src/ceb58993-7eb1-43a6-bb15-4121e18b67c0/scratchpad/audit_inputs/'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 2000); pd.set_option('display.max_columns', 50)
MON = ['25-01', '25-02', '25-06', '25-07', '26-01', '26-02', '26-06', '26-07']


def keys(which):
    p = R + ('cache/training.parquet' if which == 'tr' else 'cache_final/ranking.parquet')
    k = pd.read_parquet(p, columns=['MVT_ID_mvt', 'PHASE_mvt', 'APT', 'MVT_TIME_UTC_mvt'], filters=[('PHASE_mvt', '==', 'DEP')])
    k = k.reset_index(drop=True)
    k['ym'] = k.MVT_TIME_UTC_mvt.dt.strftime('%y-%m')
    k['day'] = k.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    k['APT'] = k.APT.astype(str)
    return k[['MVT_ID_mvt', 'APT', 'ym', 'day']]


KT, KR = keys('tr'), keys('rk')


def summarize(name, ftr, frk, cols, out):
    """ftr/frk aligned with KT/KR rows."""
    m = pd.concat([pd.concat([KT, ftr[cols].reset_index(drop=True)], axis=1),
                   pd.concat([KR, frk[cols].reset_index(drop=True)], axis=1)], ignore_index=True)
    m = m[m.ym.isin(MON)]
    rows = []
    for c in cols:
        x = pd.to_numeric(m[c], errors='coerce') if m[c].dtype.kind not in 'ifb' else m[c].astype(float)
        g = x.groupby(m.ym)
        miss = x.isna().groupby(m.ym).mean()
        q = g.quantile([.1, .5, .9]).unstack()
        r = pd.DataFrame({'miss': miss, 'p10': q[.1], 'p50': q[.5], 'p90': q[.9]}).reindex(MON)
        r['col'] = c
        rows.append(r.reset_index())
    t = pd.concat(rows)
    wide = t.pivot(index='col', columns='ym', values=['miss', 'p50'])
    print(f'\n===== {name}: missing share by month =====', file=out)
    print(wide['miss'][MON].round(4).to_string(), file=out)
    print(f'\n===== {name}: median by month =====', file=out)
    print(wide['p50'][MON].round(2).to_string(), file=out)
    print(f'\n===== {name}: p10/p90 by month =====', file=out)
    print(t.pivot(index='col', columns='ym', values='p10')[MON].round(1).to_string(), file=out)
    print(t.pivot(index='col', columns='ym', values='p90')[MON].round(1).to_string(), file=out)
    # flags: 2026 month whose missing share exceeds 2025 counterpart by > 3 pts and other 2026 months mean by > 3 pts
    fl = []
    for c in cols:
        mm = wide['miss'].loc[c]
        for mo in ['01', '02', '06', '07']:
            a, b = mm['26-' + mo], mm['25-' + mo]
            oth = np.mean([mm['26-' + o] for o in ['01', '02', '06', '07'] if o != mo])
            if a - b > 0.02 and a - oth > 0.02:
                fl.append((c, '26-' + mo, round(b, 4), round(a, 4), round(oth, 4)))
    print(f'\n{name} MISSING FLAGS (col, month, 2025, 2026, other2026):', fl, file=out)
    # per airport x month missing share for cols, flagged
    m2 = m.copy()
    apt_fl = []
    for c in cols:
        x = m2[c].isna()
        tb = x.groupby([m2.APT, m2.ym]).mean().unstack().reindex(columns=MON)
        for apt in tb.index:
            for mo in ['01', '02', '06', '07']:
                a, b = tb.loc[apt, '26-' + mo], tb.loc[apt, '25-' + mo]
                oth = np.nanmean([tb.loc[apt, '26-' + o] for o in ['01', '02', '06', '07'] if o != mo])
                if a - b > 0.05 and a - oth > 0.05:
                    apt_fl.append((c, apt, '26-' + mo, round(b, 3), round(a, 3), round(oth, 3)))
    print(f'{name} APT MISSING FLAGS (col, apt, month, 2025, 2026, other2026):', file=out)
    for f in apt_fl:
        print('   ', f, file=out)
    return m


out = open(O + 'a3.out', 'w')
# ---- v11
v11c = ['off_aobt3', 'off_sched', 'hour', 'aobt_missing', 'nm_delay', 'off_iobt', 'lobt_m_iobt', 'eobt_m_lobt', 'a3_m_eobt',
        'a3_m_lobt', 'eobt_m_sched', 'iobt_m_sched', 'pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt', 'sl_to', 'sl_after_sched',
        'sl_same_typ', 'sl_same_op', 'si_before_sched', 'si_same_typ', 'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to',
        'sd_next_to', 'sd_prev_to', 'k_op', 'k_stand', 'k_ades']
ftr = pd.read_parquet(R + 'cache/v11/feat_training.parquet', columns=['MVT_ID_mvt'] + v11c)
frk = pd.read_parquet(R + 'cache_final/v11/feat_ranking.parquet', columns=['MVT_ID_mvt'] + v11c)
assert (ftr.MVT_ID_mvt.to_numpy() == KT.MVT_ID_mvt.to_numpy()).all() and (frk.MVT_ID_mvt.to_numpy() == KR.MVT_ID_mvt.to_numpy()).all()
summarize('v11', ftr, frk, v11c, out)
# initial vs final on the same Jan/Jul rows
fin = pd.read_parquet(R + 'cache/v11/feat_ranking.parquet', columns=['MVT_ID_mvt'] + v11c).set_index('MVT_ID_mvt')
ff = frk.set_index('MVT_ID_mvt').loc[fin.index]
print('\n===== v11 initial vs final (same Jan/Jul rows): share of rows differing =====', file=out)
for c in v11c:
    a, b = fin[c].to_numpy(dtype=float), ff[c].to_numpy(dtype=float)
    diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    print(f'  {c:18s} {diff.mean():.5f}  nan init {np.isnan(a).mean():.4f} final {np.isnan(b).mean():.4f}', file=out)
del ftr, frk, fin, ff
out.flush()

# ---- adsb.lol
atr = pd.read_parquet(R + 'cache/adsb_lol/features_training.parquet').drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(KT.MVT_ID_mvt.to_numpy())
ark = pd.read_parquet(R + 'cache_final/adsb_lol/features_ranking.parquet').drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(KR.MVT_ID_mvt.to_numpy())
ac = [c for c in ark.columns if c.startswith('adsb_')]
print('\nadsb ranking ids missing from file:', int(ark.adsb_day_ok.isna().sum()), file=out)
for c in ac:
    if atr[c].dtype.kind not in 'ifb':
        print('non-numeric adsb col', c, atr[c].dtype, file=out)
        atr[c] = atr[c].astype('category').cat.codes.where(atr[c].notna())
        ark[c] = ark[c].astype('category').cat.codes.where(ark[c].notna())
m = summarize('adsb', atr, ark, ac, out)
# per day coverage: day_ok, matched, off_best
m['ok'] = m.adsb_day_ok.eq(1)
m['mat'] = m.adsb_matched.eq(1)
m['best'] = m.adsb_off_best.notna()
d = m.groupby(['APT', 'ym', 'day']).agg(n=('ok', 'size'), ok=('ok', 'mean'), mat=('mat', 'mean'), best=('best', 'mean')).reset_index()
med = d.groupby(['APT', 'ym'])[['mat', 'best']].transform('median')
d['mat_med'], d['best_med'] = med.mat, med.best
print('\n===== adsb per-airport/month means (matched / off_best) =====', file=out)
print(d.groupby(['APT', 'ym']).mat.mean().unstack().round(3).to_string(), file=out)
print(d.groupby(['APT', 'ym']).best.mean().unstack().round(3).to_string(), file=out)
bad = d[(d.ok < 0.99) | (d.mat < 0.5 * d.mat_med) | (d.best < 0.5 * d.best_med)]
print('\n===== adsb bad days (day_ok<0.99 or matched/best < half the airport-month median) =====', file=out)
print(bad.round(3).to_string(), file=out)
# initial vs final on Jan/Jul
ini = pd.read_parquet(R + 'cache/adsb_lol/features_ranking.parquet').drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
fr = ark.loc[ark.index.isin(ini.index)]
ini = ini.loc[fr.index]
print('\n===== adsb initial vs final (same Jan/Jul rows) =====', file=out)
for c in ['adsb_day_ok', 'adsb_matched', 'adsb_off_best', 'adsb_to_dev', 'adsb_n_pts', 'adsb_apt_day_ground']:
    a, b = pd.to_numeric(ini[c], errors='coerce').to_numpy(dtype=float), pd.to_numeric(fr[c], errors='coerce').to_numpy(dtype=float)
    diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    print(f'  {c:22s} differ {diff.mean():.4f}  notna init {np.isfinite(a).mean():.4f} final {np.isfinite(b).mean():.4f}', file=out)
ini_ids = set(ini.index)
dd = KR.set_index('MVT_ID_mvt').loc[fr.index]
a = pd.to_numeric(ini['adsb_off_best'], errors='coerce'); b = pd.to_numeric(fr['adsb_off_best'], errors='coerce')
x = pd.DataFrame({'day': dd.day.to_numpy(), 'ini': a.notna().to_numpy(), 'fin': b.notna().to_numpy()})
g = x.groupby('day').mean()
print('days where off_best coverage changed by > 5 pts between initial and final:', file=out)
print(g[(g.ini - g.fin).abs() > 0.05].round(3).to_string(), file=out)
out.close()
