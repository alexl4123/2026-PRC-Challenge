import json, collections, numpy as np
rows = json.load(open('lb_rows.json'))
N=344841
by = collections.defaultdict(list)
for r in rows: by[r['teamName']].append(r)
for t in by: by[t].sort(key=lambda r: r['processedAt'])
best = {t: min(r['score'] for r in v) for t,v in by.items()}
rank = sorted(best.items(), key=lambda kv: kv[1])
SINCE='2026-09-27'
print(f"{'rk':>3} {'team':24s} {'best':>7} {'n':>4} {'n10d':>4} {'nb':>3} {'big1':>6} {'d10':>6} {'gapmed':>6} {'gap>5':>5} {'gap>15':>6} {'burst':>5} {'midn':>4}")
out=[]
for i,(t,s) in enumerate(rank[:40],1):
    v=by[t]; run=float('inf'); gaps=[]; nb=0; steps=[]; n10=0
    pre=[r['score'] for r in v if r['processedAt']<SINCE]; run=min(pre) if pre else float('inf'); start=run
    times=[]
    for r in v:
        if r['processedAt']<SINCE: continue
        n10+=1; times.append(r['processedAt'])
        if r['score']<run:
            if run<float('inf'): steps.append(run-r['score'])
            nb+=1; run=r['score']
        else: gaps.append(r['score']-run)
    # bursts: >=3 submissions within 10 minutes
    import datetime as dt
    ts=[dt.datetime.fromisoformat(x[:19]) for x in times]
    burst=sum(1 for j in range(len(ts)-2) if (ts[j+2]-ts[j]).total_seconds()<=600)
    midn=sum(1 for x in ts if x.hour==0 and x.minute<20)
    big1=max(steps) if steps else 0
    d10=(start-s) if start<float('inf') else float('nan')
    g=np.array(gaps) if gaps else np.array([np.nan])
    print(f"{i:3d} {t:24s} {s:7.2f} {len(v):4d} {n10:4d} {nb:3d} {big1:6.2f} {d10:6.2f} {np.nanmedian(g):6.2f} {int((g>5).sum()):5d} {int((g>15).sum()):6d} {burst:5d} {midn:4d}")
