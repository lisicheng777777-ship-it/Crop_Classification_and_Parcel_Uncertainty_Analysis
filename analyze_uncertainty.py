"""Reproduce numerical uncertainty/risk diagnostics from fixed-test pixel probabilities."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cropaggregation.aggregation import PC, aggregate, choose_d0
from cropaggregation.uncertainty import _summary, _curve

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/fixed')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/uncertainty')
    p.add_argument('--year',choices=['2021_2022','2025_2026'],default='2025_2026')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[];datasets=[]
    paths=sorted((a.input/'runs'/a.year).glob('seed_*/T0_standard/*/source_test_pixel_probabilities.csv'))
    if not paths:raise FileNotFoundError('No fixed-test probabilities under '+str(a.input))
    for path in paths:
        pixel=pd.read_csv(path)
        d0,_=choose_d0(pd.read_csv(path.parent/'source_validation_pixel_probabilities.csv'))
        for method in ['M0','M1','M2']:
            parcel=aggregate(pixel,method,d0)
            probabilities=parcel[PC].to_numpy(float)
            parcel['confidence_uncertainty']=1-probabilities.max(1)
            parcel['entropy_norm']=-(probabilities*np.log(np.clip(probabilities,1e-15,1))).sum(1)/np.log(3)
            parcel['error']=(probabilities.argmax(1)!=parcel.y_true.to_numpy()).astype(int)
            scores=['confidence_uncertainty','entropy_norm']
            # Archived MC mutual information is used only if actually present.
            if 'epistemic' in pixel and pixel.epistemic.notna().all():
                parcel['mean_pixel_epistemic']=parcel.polygon_id.map(pixel.groupby('polygon_id').epistemic.mean())
                scores.append('mean_pixel_epistemic')
            tags={'year':a.year,'model':path.parent.name,'seed':int(path.parents[2].name.replace('seed_','')),'method':method}
            for score in scores:
                rows.append(dict(**tags,score=score,**_summary(parcel.error,parcel[score])))
                coverage,risk,generalized=_curve(parcel.error,parcel[score])
                curves.append(pd.DataFrame(dict(**tags,score=score,coverage=coverage,risk=risk,generalized_risk=generalized)))
            datasets.append(parcel.assign(**tags))
    pd.DataFrame(rows).to_csv(a.output/'risk_metrics.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(a.output/'risk_coverage.csv',index=False)
    pd.concat(datasets,ignore_index=True).to_csv(a.output/'parcel_uncertainty.csv',index=False)
    print('Uncertainty analysis complete:',len(paths),'configurations',flush=True)

if __name__=='__main__':main()
