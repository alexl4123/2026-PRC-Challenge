"""Recording-sequence check: do MVT_ID / FLIGHT_ID orders follow BLOCK, MVT or SCHED order (2025), and what is the 2026 order like?"""
import numpy as np, pandas as pd
H = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/hunt/'
for which in ('training', 'ranking'):
    t = pd.read_parquet(H + f'stand_ev_{which}.parquet', columns=['id', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'APT', 'mvt', 'blk', 'sch', 'lobt'])
    t['fid'] = pd.to_numeric(t.FLIGHT_ID_mvt, errors='coerce')
    print('=====', which, 'id dtype', t.id.dtype, 'id range', t.id.min(), t.id.max(), 'unique', t.id.nunique() == len(t))
    for ph in ('DEP', 'ARR'):
        d = t[t.PHASE_mvt == ph]
        for idcol in ('id', 'fid'):
            s = d.dropna(subset=[idcol]).sort_values(['APT', idcol])
            same = s.APT.values[1:] == s.APT.values[:-1]
            msg = []
            for col in ('blk', 'mvt', 'sch', 'lobt'):
                v = s[col].to_numpy(); dv = v[1:] - v[:-1]; ok = np.isfinite(dv) & same
                if ok.sum() == 0:
                    continue
                msg.append(f'{col}: nondecr {np.mean(dv[ok] >= 0):.4f} |d| med {np.median(np.abs(dv[ok])):.0f}')
            # global (all APT) order
            g = d.dropna(subset=[idcol]).sort_values(idcol)
            v = g.mvt.to_numpy(); dv = v[1:] - v[:-1]
            print(f'{ph} by {idcol} within APT: ' + ' | '.join(msg) + f' || global mvt nondecr {np.mean(dv >= 0):.4f}')
    # how are ids laid out? first rows
    d = t[t.PHASE_mvt == 'DEP'].sort_values('id').head(12)
    print(d[['id', 'fid', 'APT', 'mvt', 'blk', 'sch']].assign(mvt=lambda x: pd.to_datetime(x.mvt, unit='s'), blk=lambda x: pd.to_datetime(x.blk, unit='s'), sch=lambda x: pd.to_datetime(x.sch, unit='s')).to_string())
