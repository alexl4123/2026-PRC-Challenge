import numpy as np, pandas as pd, sys
z=np.load(sys.argv[1]+'/base.npz', allow_pickle=True)
y,B=z['y'],z['B']; e=y-B
grp=np.where(z['P'],'P',np.where(z['D'],'D','ABC'))
df=pd.DataFrame(dict(e=e,e2=e**2,apt=z['apt'],grp=grp,month=z['month'],hour=z['hour'],r4=z['route4'],tail=y>7200,
   olb=pd.cut(z['off_lobt'],[-1e9,0,600,1200,1800,2400,3000,3606,5000,1e9]).astype(str), big=np.abs(e)>3600))
N=len(df); T=df.e2.sum()
print('total RMSE %.3f, mean resid %.2f' % (np.sqrt(T/N), e.mean()))
def tab(keys):
    g=df.groupby(keys).agg(n=('e','size'),mean=('e','mean'),sse=('e2','sum'))
    g['share_sse']=g.sse/T; g['rmse']=np.sqrt(g.sse/g.n); g['bias2_share']=g.n*g['mean']**2/T
    return g.drop(columns='sse')
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
print(tab(['grp'])); print(tab(['grp','tail'])); print(tab(['grp','big']))
print(tab(['r4']))
print(tab(['apt','grp']).round(4))
print(tab(['olb']).round(4))
g=tab(['apt','grp','month']); print('sum bias2 share apt x grp x month: %.4f' % g.bias2_share.sum())
g2=tab(['apt','grp']); print('sum bias2 share apt x grp: %.4f' % g2.bias2_share.sum())
g3=tab(['apt','grp','hour']); print('sum bias2 share apt x grp x hour: %.4f' % g3.bias2_share.sum())
# month-to-month stability of airport x group bias (ABC)
pv=g.reset_index(); pv=pv[pv.grp=='ABC'].pivot(index='apt',columns='month',values='mean').round(1); print(pv)
pv=g.reset_index(); pv=pv[pv.grp=='D'].pivot(index='apt',columns='month',values='mean').round(0); print(pv)
# top rows concentration
s=np.sort(df.e2.values)[::-1]; print('SSE share of top 100/1000/10000 rows: %.3f %.3f %.3f' % (s[:100].sum()/T, s[:1000].sum()/T, s[:10000].sum()/T))
