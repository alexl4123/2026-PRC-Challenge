"""Flight-identity residual TE: does the SAME flight's residual in OTHER months predict this month's?"""
from common import *
d = load_dep(['APT', 'FLIGHT_mvt', 'ADES_mvt', 'STAND_mvt', 'AIRCRAFT_OPERATOR_flt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
              'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt'])
y = d.TAXITIME_SEC_mvt.to_numpy(); M = d.MVT_TIME_UTC_mvt.to_numpy(); S = d.SCHED_TIME_UTC_mvt.to_numpy()
A3 = d.AOBT_3_flt.to_numpy()
v6, v7, b = preds()
P = np.isnan(A3) & (M - S > 7200)
# approx v8 outside P: average of v7 (== v6 outside P) and 3-seed net mixture
v8 = np.where(P, v7, 0.5 * v7 + 0.5 * b[1])
month = pd.to_datetime(M, unit='s').month.to_numpy()
sc = np.isin(month, [1, 7])
apt = S_(d.APT)
cs = S_(d.FLIGHT_mvt)
sched_slot = ((S % 86400) // 1800).astype(int).astype(str).astype(object)
dow = pd.to_datetime(M, unit='s').dayofweek.astype(str).to_numpy().astype(object)
keys = {
    'apt|stand (control, model has it)': K(apt, S_(d.STAND_mvt)),
    'apt|operator (control)': apt + '|' + S_(d.AIRCRAFT_OPERATOR_flt),
    'apt|callsign': apt + '|' + cs,
    'apt|callsign|dow': apt + '|' + cs + '|' + dow,
    'apt|ADES': apt + '|' + S_(d.ADES_mvt),
    'apt|ADES|runway': apt + '|' + S_(d.ADES_mvt) + '|' + S_(d.RUNWAY_mvt),
    'apt|stand|sched30min': apt + '|' + S_(d.STAND_mvt) + '|' + sched_slot,
    'apt|callsign|stand': apt + '|' + cs + '|' + S_(d.STAND_mvt),
}
print('rows', len(y), 'P', P.sum(), ' v6', rmse(y, v6), ' v8~', rmse(y, v8))
W = 20.0
for base_name, base in [('v6', v6), ('v8~', v8)]:
    r = y - base
    src = (~P) & (y <= 7200)
    rc = np.clip(r, -3000, 3000)
    print(f'\n##### base {base_name}: RMSE all {rmse(y, base):.3f}  outsideP {rmse(y[~P], base[~P]):.3f}  scored {rmse(y[sc], base[sc]):.3f}')
    for kn, k in keys.items():
        codes, uniq = pd.factorize(k)
        nk = len(uniq)
        # per (key, month) sums from source rows
        Sm = np.zeros((nk, 13)); Nm = np.zeros((nk, 13))
        np.add.at(Sm, (codes[src], month[src]), rc[src]); np.add.at(Nm, (codes[src], month[src]), 1)
        St, Nt = Sm.sum(1), Nm.sum(1)
        s_ex = St[codes] - Sm[codes, month]; n_ex = Nt[codes] - Nm[codes, month]
        te = s_ex / (n_ex + W)                      # shrink to 0 (residual already centred by the model)
        new = base + np.where(P, 0.0, te)
        cov = (n_ex > 0).mean()
        rho = pd.Series(te[src & sc]).corr(pd.Series(r[src & sc]), method='spearman')
        d_all = rmse(y, new) - rmse(y, base)
        d_out = rmse(y[~P], new[~P]) - rmse(y[~P], base[~P])
        d_sc = rmse(y[sc], new[sc]) - rmse(y[sc], base[sc])
        bb = boot(y[~P], new[~P], base[~P], n=200)
        print(f'{kn:36s} keys {nk:7d} cov {cov:.3f} rho_sc {rho:+.3f} | d_all {d_all:+.3f} d_outP {d_out:+.3f} [{bb[2]:+.3f},{bb[3]:+.3f}] d_scored {d_sc:+.3f}')
print('peak GB', peak())
