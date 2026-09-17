"""Compute numeric results from predictions: no figures, Word or Excel reports."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,wilcoxon
from cropaggregation.aggregation import aggregate,metrics,choose_d0,brier_losses,canonical,PC
MODELS={'XGBoost','LSTM','TempCNN','VanillaTransformer','STSMamba'}
def summarize_deltas(df,out):
    cols=['brier_improvement','nll_improvement','delta_macro_f1','delta_oa']
    groups=[x for x in ['direction','method'] if x in df]
    rows=[]
    for key,g in df.groupby(groups[0] if len(groups)==1 else groups):
        vals=key if isinstance(key,tuple) else (key,)
        for col in cols:
            v=g[col].to_numpy();stat,p=wilcoxon(v,zero_method='wilcox',method='approx') if np.any(v) else (0.,1.)
            rows.append(dict(zip(groups,vals),metric=col,n=len(v),mean=v.mean(),sd=v.std(ddof=1) if len(v)>1 else np.nan,positive=int((v>0).sum()),negative=int((v<0).sum()),wilcoxon_statistic=float(stat),p_value=float(p)))
    pd.DataFrame(rows).to_csv(out/'paired_summary.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['fixed','spatial','temporal'],required=True)
    p.add_argument('--input',required=True,help='fixed: directory containing runs/; spatial/temporal: generalization output root')
    p.add_argument('--output',required=True)
    p.add_argument('--year',default='2025_2026')
    a=p.parse_args();base=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.experiment!='fixed':
        from cropaggregation import generalization as g
        rows=[]
        if a.experiment=='spatial':
            paths=list((base/'spatial_cv'/a.year).glob('*/seed_*/spatial_oof_pixel_predictions_with_distance.csv'))
            for path in paths:
                model=path.parent.parent.name;seed=path.parent.name
                if model not in MODELS:continue
                q=g._crossfit_distance(pd.read_csv(path),out/'runs'/model/seed)
                q['model']=model;q['seed']=seed;rows.append(q)
        else:
            paths=list((base/'temporal').glob('*/*/seed_*/target_pixel_predictions_with_distance.csv'))
            for path in paths:
                if path.parent.parent.name not in MODELS:continue
                d0=float(pd.read_csv(path.parent/'frozen_source_d0.csv').selected_d0_one_se.iloc[0])
                pixel=canonical(pd.read_csv(path))
                for method in ['M0','M2']:
                    pred=aggregate(pixel,method,d0)
                    rows.append(pd.DataFrame([dict(direction=path.parent.parent.parent.name,model=path.parent.parent.name,seed=path.parent.name,method=method,**metrics(pred))]))
        if not rows:raise FileNotFoundError('No matching predictions under '+str(base))
        result=pd.concat(rows,ignore_index=True);result.to_csv(out/'metrics.csv',index=False)
        keys=[x for x in ['direction','model','seed'] if x in result]
        baseline=result[result.method.isin(['M0','M0_equal'])]
        candidate=result[~result.method.isin(['M0','M0_equal'])]
        paired=baseline.merge(candidate,on=keys,suffixes=('_0','_2'),validate='one_to_one')
        change=paired[keys].copy();change['method']='M2'
        change['brier_improvement']=paired.brier_0-paired.brier_2;change['nll_improvement']=paired.nll_0-paired.nll_2
        change['delta_macro_f1']=paired.macro_f1_2-paired.macro_f1_0
        oa='accuracy' if 'accuracy_0' in paired else 'oa';change['delta_oa']=paired[oa+'_2']-paired[oa+'_0']
        change.to_csv(out/'paired_improvements.csv',index=False);summarize_deltas(change,out)
        print('Analysis complete:',out);return
    paths=list((base/'runs'/a.year).glob('seed_*/T0_standard/*/source_test_pixel_probabilities.csv'))
    if not paths:raise FileNotFoundError('No fixed-test pixel predictions found')
    run_rows=[];delta=[];pixel_rows=[];effects=[];curves=[];diagnostics=[];common_rows=[];geometry_reference=None
    for path in paths:
        model=path.parent.name;seed=int(path.parents[2].name.replace('seed_',''))
        if model not in MODELS:continue
        pixel=pd.read_csv(path);val=pd.read_csv(path.parent/'source_validation_pixel_probabilities.csv')
        geometry=pixel[['polygon_id','distance_px','y_true']].reset_index(drop=True)
        if geometry_reference is None:geometry_reference=geometry
        elif not geometry.equals(geometry_reference):raise ValueError('Pixel geometry/order differs across fixed-test configurations')
        d0,curve=choose_d0(val);curve['model']=model;curve['seed']=seed;curves.append(curve)
        preds={method:aggregate(pixel,method,d0) for method in ['M0','M1','M2']}
        n=len(preds['M0']);pixel_rows.append(dict(model=model,seed=seed,**metrics(pixel)))
        for method,pred in preds.items():run_rows.append(dict(model=model,seed=seed,method=method,d0=d0,n_parcels=len(pred),coverage=len(pred)/n,**metrics(pred)))
        for method in ['M1','M2']:
            x=preds[method];m0=preds['M0'][preds['M0'].polygon_id.isin(x.polygon_id)];aa=metrics(m0);bb=metrics(x)
            if method=='M1':common_rows.append(dict(model=model,seed=seed,n_parcels=len(m0),**aa))
            delta.append(dict(model=model,seed=seed,method=method,brier_improvement=aa['brier']-bb['brier'],nll_improvement=aa['nll']-bb['nll'],delta_macro_f1=bb['macro_f1']-aa['macro_f1'],delta_oa=bb['oa']-aa['oa']))
        pred=preds['M0'].copy();pred['brier_improvement']=brier_losses(pred)-brier_losses(preds['M2'])
        pred['M1_covered']=pred.polygon_id.isin(preds['M1'].polygon_id);pred['model']=model;pred['seed']=seed;effects.append(pred[['polygon_id','n_pixels','brier_improvement','M1_covered','model','seed']])
        # Equal parcel weighting within relative-position bins.
        for pid,g in pixel.groupby('polygon_id'):
            if len(g)<2:continue
            q=(g.distance_px.rank(method='average')-1)/(len(g)-1);bins=np.minimum((q*5).astype(int),4)
            prob=g[PC].to_numpy();truth=g.y_true.to_numpy(int)
            z=pd.DataFrame({'bin':bins.to_numpy(),'error':(prob.argmax(1)!=truth).astype(float),'brier':brier_losses(g),'tcp':prob[np.arange(len(g)),truth],'entropy':-(prob*np.log(np.clip(prob,1e-15,1))).sum(1)/np.log(3)})
            z=z.groupby('bin',as_index=False).mean();z['polygon_id']=pid;z['model']=model;z['seed']=seed;diagnostics.append(z)
    pd.DataFrame(run_rows).to_csv(out/'fixed_metrics.csv',index=False)
    pd.DataFrame(common_rows).to_csv(out/'M0_common_support_metrics.csv',index=False)
    pd.DataFrame(delta).to_csv(out/'paired_improvements.csv',index=False)
    summarize_deltas(pd.DataFrame(delta),out)
    pd.concat(curves).to_csv(out/'d0_validation_curves.csv',index=False)
    px=pd.DataFrame(pixel_rows);px.to_csv(out/'pixel_metrics.csv',index=False);px.groupby('model')[['brier','nll','macro_f1','oa']].agg(['mean','std']).to_csv(out/'pixel_model_summary.csv')
    dg=pd.concat(diagnostics);dg.groupby(['model','seed','bin'])[['error','brier','tcp','entropy']].mean().reset_index().to_csv(out/'boundary_diagnostics.csv',index=False)
    e=pd.concat(effects);e.to_csv(out/'parcel_effects_by_run.csv',index=False)
    # Geometry is common across configurations; validate before using one input.
    geo=pixel.groupby('polygon_id').distance_px.agg(['size',lambda x:np.std(x,ddof=0)/max(x.max(),1e-12)]);geo.columns=['geometry_n','distance_std_norm']
    parcels=e.groupby('polygon_id').agg({'n_pixels':'first','brier_improvement':'mean','M1_covered':'first'}).join(geo)
    assert (parcels.n_pixels==parcels.geometry_n).all()
    parcels.to_csv(out/'parcel_mean_effects.csv')
    rng=np.random.default_rng(20260914);rows=[]
    for lo,hi in [(1,2),(3,4),(5,6),(7,9),(10,10**9)]:
        z=parcels[parcels.n_pixels.between(lo,hi)];v=z.brier_improvement.to_numpy();n=len(v)
        bs=rng.choice(v,(5000,n)).mean(1) if n else np.array([np.nan])
        rows.append(dict(min_pixels=lo,max_pixels=hi,n=n,M1_coverage=z.M1_covered.mean(),mean_brier_improvement=z.brier_improvement.mean(),ci95_low=np.nan if not n else np.quantile(bs,.025),ci95_high=np.nan if not n else np.quantile(bs,.975)))
    pd.DataFrame(rows).to_csv(out/'size_analysis.csv',index=False)
    small=parcels[parcels.n_pixels<=9];rs,pvalue=spearmanr(small.distance_std_norm,small.brier_improvement)
    (out/'distance_correlation.json').write_text(json.dumps(dict(n=len(small),r_s=float(rs),p_value=float(pvalue)),indent=2))
    print('Analysis complete:',out)
if __name__=='__main__':main()
