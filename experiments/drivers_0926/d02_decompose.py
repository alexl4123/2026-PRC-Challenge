"""Drivers 09-26, step 2: decompose the v17-structure 2025 OOF squared error, and compare the 2026 ranking set.

Reads cache/drivers_0926/rows_{training,ranking}.parquet (d01_rows.py), cache/w2/v15s_oof.npy (pre-ADS-B
structure, for the with/without-ADS-B contrast on the SAME rows) and cache/tfm_w4/w4_oof.npz (P-zone
probabilities used by v17's P branch). Writes CSV tables + summary.json to cache/drivers_0926/ and prints them.

Error MECHANISM (post hoc, uses the LABEL; for diagnosis only, never a feature):
  day      y > 12 h and |y - off_sched| > 600        block written a day (or most of a day) early
  sched2h  y > 2 h and |y - off_sched| <= 120        BLOCK = SCHED on a long-delayed flight (stamped)
  early    y > 2 h, LIRF, 3,600 <= y - off_sched <= 7,800   BLOCK 1-2 h BEFORE SCHED (Rome mode, IDEAS 25)
  long     other y > 2 h
  Pfalse   zone P (no NM record, off_sched > 2 h) and y <= 2 h: a real (late) pushback
  schedB   y <= 2 h, |y - off_sched| <= 120, off_sched > 1,200: stamped-at-schedule SIGNATURE in the bulk
           (ambiguous: an on-time pushback with a 20+ min taxi has it too)
  ord30 / ord60 / ord120   everything else, by label band <= 30 min / 30-60 / 60-120

    python experiments/drivers_0926/d02_decompose.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import peak_rss_gb

OUT = CACHE / 'drivers_0926'
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
MECH_ORDER = ['ord30', 'ord60', 'ord120', 'schedB', 'Pfalse', 'sched2h', 'early', 'long', 'day']
SUMMARY = {}


def mechanism(r):
    y, so = r.y.to_numpy(), r.off_sched.to_numpy()
    d = y - so; ad = np.abs(d); lirf = (r.apt == 'LIRF').to_numpy(); P = (r.route == 'P').to_numpy()
    m = np.select(
        [(y > 43200) & ~(ad <= 600), (y > 7200) & (ad <= 120), (y > 7200) & lirf & (d >= 3600) & (d <= 7800),
         y > 7200, P & (y <= 7200), (y <= 7200) & (ad <= 120) & (so > 1200), y <= 1800, y <= 3600],
        ['day', 'sched2h', 'early', 'long', 'Pfalse', 'schedB', 'ord30', 'ord60'], 'ord120')
    return m


def adsb_status(r):
    return np.where(r.adsb_off, 'off-block seen', np.where(r.adsb_ground, 'ground only', 'no ground'))


def share_table(r, e2, keys, scope, name):
    tot = e2[scope].sum(); n = scope.sum(); rm = np.sqrt(tot / n)
    g = pd.DataFrame({'k': keys[scope], 'e2': e2[scope]}).groupby('k', observed=True)['e2'].agg(['size', 'sum'])
    g.columns = ['rows', 'sse']
    g['row_pct'] = 100 * g.rows / n; g['sse_pct'] = 100 * g.sse / tot; g['rmse'] = np.sqrt(g.sse / g.rows)
    g['halffix'] = rm - np.sqrt((tot - 0.75 * g.sse) / n)       # all-flights RMSE drop if the cell's RMSE halved
    g['zerofix'] = rm - np.sqrt((tot - g.sse) / n)              # ... if the cell were predicted perfectly (oracle)
    g = g.sort_values('sse', ascending=False)
    g.drop(columns='sse').to_csv(OUT / f'{name}.csv')
    return g, rm


def main():
    t0 = time.time()
    r = pd.read_parquet(OUT / 'rows_training.parquet'); k = pd.read_parquet(OUT / 'rows_ranking.parquet')
    v15 = np.load(CACHE / 'w2' / 'v15s_oof.npy').astype(float)
    y, f = r.y.to_numpy(), r.pred.to_numpy(); e2 = (y - f) ** 2; e15 = (y - v15) ** 2
    S = r.month.isin([1, 7]).to_numpy(); ALL = np.ones(len(r), bool)
    r['mech'] = mechanism(r); r['adsb'] = adsb_status(r); k['adsb'] = adsb_status(k)
    r['band'] = np.select([y <= 1800, y <= 3600, y <= 7200], ['<=30m', '30-60m', '1-2h'], '>2h')
    apt = r.apt.astype(str).to_numpy(); route = r.route.astype(str).to_numpy()
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        SUMMARY[f'rmse_{sc_name}'] = float(np.sqrt(e2[sc].mean()))
    print(f"RMSE 12m {SUMMARY['rmse_12m']:.2f} | scored (Jan+Jul 2025) {SUMMARY['rmse_scored']:.2f}")

    # ---- 1. mechanism ----------------------------------------------------------------------------------
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        g, rm = share_table(r, e2, r.mech.to_numpy(), sc, f't1_mech_{sc_name}')
        print(f'\n[1] MECHANISM, {sc_name} (RMSE {rm:.2f})'); print(g.drop(columns='sse').round(3).to_string())
        SUMMARY[f'mech_{sc_name}'] = g[['rows', 'sse_pct', 'rmse', 'halffix', 'zerofix']].round(4).to_dict('index')

    # ---- 2. airport x route x band ---------------------------------------------------------------------
    key = np.char.add(np.char.add(np.char.add(apt, '|'), route), np.char.add('|', r.band.to_numpy().astype(str)))
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        g, rm = share_table(r, e2, key, sc, f't2_apt_route_band_{sc_name}')
        print(f'\n[2] AIRPORT x ROUTE x BAND, {sc_name}: top 20'); print(g.drop(columns='sse').head(20).round(3).to_string())
    key2 = np.char.add(np.char.add(apt, '|'), route)
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        g, rm = share_table(r, e2, key2, sc, f't2b_apt_route_{sc_name}')
    # airport x mechanism (for the heatmap)
    key3 = np.char.add(np.char.add(apt, '|'), r.mech.to_numpy().astype(str))
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        share_table(r, e2, key3, sc, f't2c_apt_mech_{sc_name}')

    # ---- 3. ADS-B coverage x airport, ordinary A/B/C rows (y <= 2 h): v15 (no ADS-B) vs v17 on the SAME rows ----
    ok = (route == 'ABC') & (y <= 7200)
    t3 = (pd.DataFrame({'apt': apt[ok], 'adsb': r.adsb.to_numpy()[ok], 'e17': e2[ok], 'e15': e15[ok]})
          .groupby(['apt', 'adsb']).agg(rows=('e17', 'size'), mse17=('e17', 'mean'), mse15=('e15', 'mean')))
    t3['rmse17'] = np.sqrt(t3.mse17); t3['rmse15'] = np.sqrt(t3.mse15); t3['gain_pct'] = 100 * (1 - t3.mse17 / t3.mse15)
    t3.to_csv(OUT / 't3_adsb_apt.csv'); print('\n[3] A/B/C y<=2h by airport x ADS-B status: v15 (pre-ADS-B) vs v17'); print(t3.round(1).to_string())
    t3s = (pd.DataFrame({'adsb': r.adsb.to_numpy()[ok & S], 'e17': e2[ok & S], 'e15': e15[ok & S]})
           .groupby('adsb').agg(rows=('e17', 'size'), mse17=('e17', 'mean'), mse15=('e15', 'mean')))
    print('   scored months pooled:'); print(np.sqrt(t3s[['mse17', 'mse15']]).assign(rows=t3s.rows).round(1).to_string())

    # label-vs-physics noise: y - adsb_taxi on push-detected, ordinary rows (robust spread) and model RMSE there
    push = ok & (r.adsb_src.to_numpy() == '2.0') & np.isfinite(r.adsb_taxi.to_numpy())
    dd = y[push] - r.adsb_taxi.to_numpy()[push]
    tn = pd.DataFrame({'apt': apt[push], 'd': dd, 'e17': e2[push]}).groupby('apt').agg(
        rows=('d', 'size'), med=('d', 'median'), mad=('d', lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))),
        rms_trim=('d', lambda v: np.sqrt(np.mean(np.clip(v - np.median(v), -600, 600) ** 2))), model_mse=('e17', 'mean'))
    tn['model_rmse'] = np.sqrt(tn.model_mse); tn.to_csv(OUT / 't3b_label_vs_adsb.csv')
    print('\n[3b] label minus ADS-B pushback taxi on push-detected ordinary rows (s): bias, robust sd, trimmed RMS; model RMSE there')
    print(tn.drop(columns='model_mse').round(1).to_string())
    four = np.isin(apt, ['EDDF', 'EDDM', 'EHAM', 'LSZH'])
    floor = float(e2[ok & four & r.adsb_off.to_numpy()].mean())
    SUMMARY['floor_mse_4apt_covered'] = floor
    print(f'   FLOOR (v17 MSE on ordinary rows with an ADS-B off-block at EDDF/EDDM/EHAM/LSZH): {floor:.0f} s^2 = RMSE {np.sqrt(floor):.1f}')

    # ---- 4. concentration --------------------------------------------------------------------------------
    conc = {}
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        s = np.sort(e2[sc])[::-1]; c = np.cumsum(s) / s.sum(); n = len(s)
        conc[sc_name] = {f'top{t}': float(c[t - 1]) for t in (10, 100, 1000, 10000, 100000)}
        conc[sc_name].update({f'top{p}pct': float(c[int(p / 100 * n) - 1]) for p in (0.1, 1, 5, 10, 50)})
        np.save(OUT / f'conc_{sc_name}.npy', c[np.unique(np.geomspace(1, n, 4000).astype(int)) - 1])
        np.save(OUT / f'conc_{sc_name}_x.npy', np.unique(np.geomspace(1, n, 4000).astype(int)) / n)
    SUMMARY['concentration'] = conc
    print('\n[4] CONCENTRATION (cumulative share of SSE in the largest-error rows)'); print(pd.DataFrame(conc).round(3).to_string())

    # ---- 5. local hour, ordinary rows ---------------------------------------------------------------------
    ordm = np.isin(r.mech.to_numpy(), ['ord30', 'ord60', 'ord120'])
    t5 = pd.DataFrame({'h': r.hour_loc.to_numpy()[ordm], 'e': e2[ordm]}).groupby('h')['e'].agg(['size', 'sum', 'mean'])
    t5['row_pct'] = 100 * t5['size'] / t5['size'].sum(); t5['sse_pct'] = 100 * t5['sum'] / t5['sum'].sum(); t5['rmse'] = np.sqrt(t5['mean'])
    t5.to_csv(OUT / 't5_hour.csv'); print('\n[5] ordinary rows by LOCAL hour'); print(t5[['size', 'row_pct', 'sse_pct', 'rmse']].round(2).to_string())

    # ---- 6. airport-days (ordinary rows), concentration + label-free day indicators --------------------
    day = pd.DataFrame({'apt': apt, 'date': r.date_loc.to_numpy(), 'e': e2, 'ordm': ordm, 'D': route != 'ABC',
                        'so': r.off_sched.to_numpy(), 'nmdel': (r.off_sched - r.off_aobt3).to_numpy()})
    gd = day[day.ordm].groupby(['apt', 'date']).agg(n=('e', 'size'), sse=('e', 'sum'))
    gd['mse'] = gd.sse / gd.n
    ind = day.groupby(['apt', 'date']).agg(n_all=('e', 'size'), shareD=('D', 'mean'), so_med=('so', 'median'),
                                          so_p90=('so', lambda v: np.nanpercentile(v, 90)))
    gd = gd.join(ind)
    apt_mse = gd.groupby('apt').apply(lambda x: x.sse.sum() / x.n.sum())
    gd['ratio'] = gd.mse / gd.index.get_level_values('apt').map(apt_mse).to_numpy()
    gd['excess'] = gd.sse - gd.n * gd.index.get_level_values('apt').map(apt_mse).to_numpy()
    gd.to_csv(OUT / 't6_airport_days.csv')
    ss = np.sort(gd.sse.to_numpy())[::-1]; cc = np.cumsum(ss) / ss.sum(); nd = len(ss)
    SUMMARY['days'] = {'n_airport_days': int(nd), 'top1pct_share': float(cc[int(0.01 * nd) - 1]),
                       'top5pct_share': float(cc[int(0.05 * nd) - 1]), 'top10pct_share': float(cc[int(0.10 * nd) - 1]),
                       'excess_pos_share': float(gd.excess.clip(lower=0).sum() / gd.sse.sum())}
    print('\n[6] ordinary-row SSE by airport-day:', SUMMARY['days'])
    print(gd.sort_values('excess', ascending=False).head(15).round(2).to_string())
    corr = gd[['ratio', 'shareD', 'so_med', 'so_p90']].corr(method='spearman')['ratio']
    print('   Spearman of the day MSE ratio with label-free day indicators:', corr.round(3).to_dict())
    SUMMARY['days']['spearman_ratio'] = corr.round(3).to_dict()

    # ---- 7. stamping signatures by airport ----------------------------------------------------------------
    so = r.off_sched.to_numpy()
    sig = (np.abs(y - so) <= 120) & (so > 1200)
    t7 = pd.DataFrame({'apt': apt, 'route': route, 'sig': sig, 'e': e2, 'day': r.mech.to_numpy() == 'day',
                       'early': r.mech.to_numpy() == 'early'}).groupby(['apt']).apply(lambda x: pd.Series({
                           'rows': len(x), 'sig_row_pct': 100 * x.sig.mean(), 'sig_sse_pct_apt': 100 * x.e[x.sig].sum() / x.e.sum(),
                           'day_rows': int(x.day.sum()), 'early_rows': int(x.early.sum())}))
    t7.to_csv(OUT / 't7_signatures.csv'); print('\n[7] stamping signatures (|y-off_sched|<=120, off_sched>1200)'); print(t7.round(2).to_string())

    # ---- 8. zone P detail (probabilities of the shipped P branch) --------------------------------------
    w = np.load(CACHE / 'tfm_w4' / 'w4_oof.npz', allow_pickle=True)
    Pidx = w['Pidx']; p = np.where(w['apt'] == 'LIRF', w['p_own'], w['p_pool'])
    pz = pd.DataFrame({'apt': np.where(w['apt'] == 'LIRF', 'LIRF', 'other'), 'mech': r.mech.to_numpy()[Pidx], 'p': p,
                       'e': e2[Pidx], 'scored': S[Pidx], 'date_change': r.date_change.to_numpy()[Pidx]})
    t8 = pz.groupby(['apt', 'mech']).agg(rows=('p', 'size'), p_mean=('p', 'mean'), sse_pct_all=('e', lambda v: 100 * v.sum() / e2.sum()),
                                        rmse=('e', lambda v: np.sqrt(v.mean())))
    t8.to_csv(OUT / 't8_P.csv'); print('\n[8] zone P by mechanism (p = shipped stamped probability)'); print(t8.round(3).to_string())
    from sklearn.metrics import roc_auc_score
    for a in ('LIRF', 'other'):
        m = (pz.apt == a) & pz.mech.isin(['sched2h', 'Pfalse'])
        print(f'   AUC stamped vs real ({a}): {roc_auc_score(pz.mech[m] == "sched2h", pz.p[m]):.3f}  n={m.sum()}')

    # ---- 9. 2026 ranking vs 2025 (label-free) ------------------------------------------------------------
    def prof(x, name):
        so = x.off_sched.to_numpy()
        g = x.groupby('apt', observed=True)
        o = pd.DataFrame({'rows': g.size(), 'pct_D': 100 * g.route.apply(lambda v: (v == 'D').mean()),
                          'pct_P': 100 * g.route.apply(lambda v: (v == 'P').mean()),
                          'pct_adsb_off': 100 * g.adsb_off.mean(), 'pct_ground': 100 * g.adsb_ground.mean(),
                          'so_med': g.off_sched.median(), 'pct_so_gt1800': 100 * g.off_sched.apply(lambda v: (v > 1800).mean()),
                          'pct_so_gt3600': 100 * g.off_sched.apply(lambda v: (v > 3600).mean()),
                          'pct_datechg': 100 * g.date_change.mean()})
        tot = pd.DataFrame({'rows': [len(x)], 'pct_D': [100 * (x.route == 'D').mean()], 'pct_P': [100 * (x.route == 'P').mean()],
                            'pct_adsb_off': [100 * x.adsb_off.mean()], 'pct_ground': [100 * x.adsb_ground.mean()],
                            'so_med': [np.nanmedian(so)], 'pct_so_gt1800': [100 * np.mean(so > 1800)],
                            'pct_so_gt3600': [100 * np.mean(so > 3600)], 'pct_datechg': [100 * x.date_change.mean()]}, index=['ALL'])
        o = pd.concat([o, tot]); o.columns = pd.MultiIndex.from_product([[name], o.columns]); return o
    r25 = r[S]
    t9 = pd.concat([prof(r25, '2025 Jan+Jul'), prof(k, '2026 Jan+Jul')], axis=1)
    t9.to_csv(OUT / 't9_2025_vs_2026.csv'); print('\n[9] 2025 Jan+Jul (labelled) vs 2026 ranking (label-free)'); print(t9.round(2).to_string())
    for mo in (1, 7):
        a, b = r[r.month == mo], k[k.month == mo]
        print(f'   month {mo}: adsb_off 2025 {100*a.adsb_off.mean():.1f}% vs 2026 {100*b.adsb_off.mean():.1f}% | P rows {int((a.route=="P").sum())} vs {int((b.route=="P").sum())} '
              f'| LIRF P {int(((a.route=="P")&(a.apt=="LIRF")).sum())} vs {int(((b.route=="P")&(b.apt=="LIRF")).sum())} | D {int((a.route=="D").sum())} vs {int((b.route=="D").sum())}')

    # ---- 10. reweighting: 2025 per-cell MSE applied to the 2026 cell mix ----------------------------------
    def cells(x):
        sob = pd.cut(x.off_sched, [-np.inf, 0, 900, 1800, 3600, 7200, np.inf], labels=False).fillna(-1).astype(int).astype(str)
        return (x.apt.astype(str) + '|' + x.route.astype(str) + '|' + x.adsb.astype(str) + '|' + sob.to_numpy()).to_numpy()
    c25, c26 = cells(r), cells(k)
    res = {}
    for label, scope, drop_day in (('12m', ALL, False), ('12m_noday', ALL, True), ('scored', S, False), ('scored_noday', S, True)):
        sc = scope & ~((r.mech.to_numpy() == 'day') if drop_day else np.zeros(len(r), bool))
        mse = pd.Series(e2[sc]).groupby(c25[sc]).mean()
        coarse = pd.Series(e2[sc]).groupby(np.array([s.rsplit('|', 1)[0] for s in c25[sc]])).mean()
        m26 = pd.Series(c26).map(mse)
        miss = m26.isna().to_numpy()
        m26[miss] = pd.Series([s.rsplit('|', 1)[0] for s in c26[miss]]).map(coarse).to_numpy()
        own = float(np.sqrt(e2[sc].mean()))
        res[label] = {'own_rmse_2025': own, 'proj_rmse_2026': float(np.sqrt(np.nanmean(m26))), 'unmatched_cells_rows': int(miss.sum())}
    SUMMARY['reweight'] = res
    print('\n[10] 2025 cell MSE (airport x route x ADS-B status x off_sched band) on the 2026 mix:'); print(pd.DataFrame(res).round(2).to_string())
    print('     live v17 = 251.12')

    # ---- 11. irreducible / recording / physical split ----------------------------------------------------
    out = {}
    for sc_name, sc in (('12m', ALL), ('scored', S)):
        tot = e2[sc].sum(); mech = r.mech.to_numpy()
        dayS = e2[sc & (mech == 'day')].sum()
        recS = e2[sc & np.isin(mech, ['sched2h', 'early', 'Pfalse', 'schedB'])].sum()
        longS = e2[sc & (mech == 'long')].sum()
        ordm_s = sc & np.isin(mech, ['ord30', 'ord60', 'ord120'])
        ordS = e2[ordm_s].sum(); floorS = ordm_s.sum() * floor
        # per-airport floor where the airport has >= 5,000 covered ordinary rows, else the 4-airport floor
        cov = ok & r.adsb_off.to_numpy()
        pf = {a: (e2[cov & (apt == a)].mean() if (cov & (apt == a)).sum() >= 5000 else floor) for a in np.unique(apt)}
        floor_apt = sum(ordm_s[apt == a].sum() * pf[a] for a in pf)
        n = sc.sum()
        out[sc_name] = {'total_rmse': float(np.sqrt(tot / n)), 'day_pct': 100 * dayS / tot, 'recording_pct': 100 * recS / tot,
                        'long_other_pct': 100 * longS / tot, 'ordinary_pct': 100 * ordS / tot,
                        'ordinary_floor4_pct': 100 * floorS / tot, 'ordinary_floorApt_pct': 100 * floor_apt / tot,
                        'rmse_without_day': float(np.sqrt((tot - dayS) / n)),
                        'rmse_if_ordinary_at_floor4': float(np.sqrt((tot - ordS + floorS) / n)),
                        'rmse_if_ordinary_at_floorApt': float(np.sqrt((tot - ordS + floor_apt) / n)),
                        'rmse_oracle_recording': float(np.sqrt((tot - recS) / n))}
    SUMMARY['split'] = out
    print('\n[11] irreducible / recording / physical split'); print(pd.DataFrame(out).round(2).to_string())

    json.dump(SUMMARY, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB | {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
