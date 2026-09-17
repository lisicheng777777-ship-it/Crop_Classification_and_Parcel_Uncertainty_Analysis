"""Numerical M0/M1/M2 aggregation; no plotting or report generation."""
import numpy as np
import pandas as pd
PC=['prob_class_0','prob_class_1','prob_class_2']
GRID=[.5,.75,1,1.5,2,3,4,5,6,8,10,12]
def canonical(df):
    return df.rename(columns={'parcel_id':'polygon_id','truth':'y_true','prob_0':PC[0],'prob_1':PC[1],'prob_2':PC[2]})
def aggregate(df,method='M0',d0=None):
    df=canonical(df);d=df.distance_px.to_numpy(float)
    if method=='M0':w=np.ones(len(df))
    elif method=='M1':w=(d>=1).astype(float)
    elif method=='M2':
        if d0 is None or d0<=0:raise ValueError('Positive d0 required for M2')
        w=np.minimum(1,d/d0)
    else:raise ValueError(method)
    rows=[]
    for pid,g in df.assign(_w=w).groupby('polygon_id'):
        if g.y_true.nunique()!=1:raise ValueError('Inconsistent parcel labels')
        weights=g._w.to_numpy()
        if weights.sum()<=1e-12:
            if method=='M1':continue
            weights=np.ones(len(g))
        p=np.average(g[PC].to_numpy(float),axis=0,weights=weights);p/=p.sum()
        rows.append([pid,int(g.y_true.iloc[0]),len(g)]+p.tolist())
    return pd.DataFrame(rows,columns=['polygon_id','y_true','n_pixels']+PC)
def brier_losses(df):
    df=canonical(df);return ((df[PC].to_numpy(float)-np.eye(3)[df.y_true.to_numpy(int)])**2).sum(1)
def metrics(df,epsilon=1e-12):
    df=canonical(df);p=df[PC].to_numpy(float);y=df.y_true.to_numpy(int);pred=p.argmax(1)
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    f=[]
    for c in range(3):
        den=(y==c).sum()+(pred==c).sum();f.append(2*((y==c)&(pred==c)).sum()/den if den else 0.)
    return dict(brier=brier_losses(df).mean(),nll=-np.log(np.clip(p[np.arange(len(y)),y],epsilon,1)).mean(),macro_f1=np.mean(f),oa=np.mean(pred==y))
def choose_d0(validation):
    rows=[]
    for d0 in GRID:
        loss=brier_losses(aggregate(validation,'M2',d0))
        rows.append(dict(d0=d0,mean_brier=loss.mean(),se_brier=loss.std(ddof=1)/np.sqrt(len(loss))))
    curve=pd.DataFrame(rows);best=curve.loc[curve.mean_brier.idxmin()]
    selected=curve.loc[curve.mean_brier<=best.mean_brier+best.se_brier,'d0'].min()
    return float(selected),curve
