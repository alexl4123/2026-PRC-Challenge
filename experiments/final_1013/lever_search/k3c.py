from lib import *
from sklearn.isotonic import IsotonicRegression
ol=Z['off_lobt']; ok=ABC&np.isfinite(ol)&(ol>3606)&~Z['proj']
e=y-B; G=B.astype(float).copy()
for m in range(1,13):
    tr=ok&(month!=m); ir=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(ol[tr],e[tr])
    s=ok&(month==m); G[s]=B[s]+ir.predict(ol[s])
G=finish(G); evaluate('K3c isotonic residual in off_lobt (>3606)',G)
gain=(y-B)**2-(y-G)**2; print('w/o rows off_lobt>8000:', r(G,~(ol>8000))-r(B,~(ol>8000)))
ir=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(ol[ok],e[ok])
for x in [3700,4200,5000,6000,7000,8000,9000,10000,12000]: print(x, round(float(ir.predict([x])[0]),1))
G3b=np.load(S+'/G_k3b.npy'); print('K3b w/o rows off_lobt>8000:', r(G3b,~(ol>8000))-r(B,~(ol>8000)), ' K3b Jan+Jul w/o >8000', r(G3b,np.isin(month,[1,7])&~(ol>8000))-r(B,np.isin(month,[1,7])&~(ol>8000)))
