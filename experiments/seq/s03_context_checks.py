"""Bet D: invariant checks and 2025 -> 2026 shift of the neighbour contexts (label-free; no model).

Builds the movement table exactly as cluster/seq_final.py does (training + ranking, training vocabulary, K=64,
+-90 min) and checks, for every departure of both files:
  1. every neighbour shares the target's (airport, UTC month) group -- no context crosses a month (ranking = Jan/Jul);
  2. |dt| <= window and no departure is its own neighbour;
  3. no departure row carries an in-block time / taxi-in (seq_net.assert_label_free, re-run here);
then compares 2025 Jan+Jul with 2026 Jan+Jul per airport: context time span (median max |dt|), arrival share of the
tokens, share of departure tokens with an adsb.lol implied taxi, and share of departure tokens with an NM proxy.

    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 nice -n 5 \
      /home/thinklex/prc2026-venv/bin/python experiments/seq/s03_context_checks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from prc.config import CACHE
from prc.io import load, peak_rss_gb
from prc.models import seq_net as S


def main():
    ftr, frk = load('training', columns=S.MV_COLS), load('ranking', columns=S.MV_COLS)
    vocab = S.vocabularies(ftr)
    n_tr = len(ftr)
    full = pd.concat([f.astype({c: object for c in f.columns if str(f[c].dtype) == 'category'}) for f in (ftr, frk)],
                     ignore_index=True)
    del ftr, frk
    adsb = pd.concat([pd.read_parquet(CACHE / 'adsb_lol' / f'features_{w}.parquet', columns=['MVT_ID_mvt'] + S.ADSB_TOK)
                      for w in ('training', 'ranking')], ignore_index=True)
    mv, _ = S.movement_table(full, adsb, vocab=vocab)
    apt_all = full.APT.astype(str).to_numpy()
    del full, adsb
    S.assert_label_free(mv)
    nb = S.build_context(mv, K=64, window=5400)
    tgt = np.flatnonzero(mv['is_dep'])
    j = np.clip(nb, 0, None); ok = nb >= 0
    same_grp = (mv['grp'][j] == mv['grp'][tgt][:, None]) | ~ok
    dt = mv['t_abs'][j] - mv['t_abs'][tgt][:, None]
    print(f'1. neighbours outside the target group: {int((~same_grp).sum())}')
    print(f'2. |dt| > 5400: {int(((np.abs(dt) > 5400) & ok).sum())}; self-neighbours: {int(((j == tgt[:, None]) & ok).sum())}')
    print(f'3. label-free assert passed; departures with an in-block time: '
          f'{int(np.isfinite(mv["T"][mv["is_dep"], S.TIME_KEYS.index("t_blk")]).sum())}')

    rk = tgt >= n_tr
    ts = pd.to_datetime(mv['t_abs'][tgt], unit='s', utc=True)
    mon = ts.month.to_numpy()
    apt = apt_all[tgt]
    span = np.where(ok, np.abs(dt), np.nan)
    is_arr = ~mv['is_dep'][j] & ok
    ia = mv['S_names'].index('adsb_off_best'); ip = mv['S_names'].index('proxy')
    dep_tok = mv['is_dep'][j] & ok
    has_adsb = dep_tok & np.isfinite(mv['S'][j, ia])
    has_nm = dep_tok & np.isfinite(mv['S'][j, ip])
    rows = []
    for x in sorted(set(apt)):
        for yr, sel in (('2025', ~rk), ('2026', rk)):
            for m in (1, 7):
                k = (apt == x) & sel & (mon == m)
                if not k.any():
                    continue
                rows.append(dict(apt=x, year=yr, month=m, deps=int(k.sum()),
                                 span_med_min=float(np.nanmedian(np.nanmax(span[k], 1)) / 60),
                                 arr_share=float(is_arr[k].sum() / ok[k].sum()),
                                 dep_tok_adsb=float(has_adsb[k].sum() / max(dep_tok[k].sum(), 1)),
                                 dep_tok_nm=float(has_nm[k].sum() / max(dep_tok[k].sum(), 1))))
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('\n4. context composition, 2025 vs 2026 (Jan / Jul):')
    print(t.round(3).to_string(index=False))
    t.to_csv(CACHE / 'seq' / 's03_context_shift.csv', index=False)
    print(f'\npeak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()
