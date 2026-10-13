import json, collections
rows=json.load(open('../lb_rows.json'))
def board(cut):
    b={}; n=collections.Counter()
    for r in rows:
        if r['processedAt']<=cut:
            n[r['teamName']]+=1
            if r['teamName'] not in b or r['score']<b[r['teamName']]: b[r['teamName']]=r['score']
    return sorted(b.items(),key=lambda kv:kv[1]), n
for cut in ['2026-09-20T00','2026-09-24T00','2026-09-27T00','2026-10-03T16:39','2026-10-03T17:39']:
    rk,n=board(cut)
    print(cut, ' | '.join(f'{i}. {t} {s:.2f} (n={n[t]})' for i,(t,s) in enumerate(rk[:5],1)))
# leader changes over time
lead=None
for r in sorted(rows,key=lambda r:r['processedAt']):
    pass
import itertools
times=sorted(set(r['processedAt'][:13] for r in rows))
prev=None
for tm in times:
    rk,_=board(tm+':59')
    if rk and rk[0][0]!=prev: print('leader from',tm,rk[0]); prev=rk[0][0]
