"""Predict fixed-test/validation probabilities using an existing fixed-run checkpoint."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/paper.json'))
    p.add_argument('--year',choices=['2025_2026','2021_2022'],required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--run-dir',required=True,help='Fixed-run folder containing Best_model, min_Max.txt and temperature')
    p.add_argument('--split',choices=['test','validation'],default='test')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=p.parse_args();c=json.loads(Path(a.config).read_text());d=c['datasets'][a.year]
    if a.model not in c['models']:p.error('Unsupported model')
    import numpy as np,pandas as pd,torch
    from cropaggregation.fixed import _load_common_split,_set_seed,BATCH_SIZES
    from cropaggregation.data import read_minMaxVal,extractValSet
    from cropaggregation.spatial_patches import extract_spatial_patches
    from cropaggregation.modeling import get_model,load_model_checkpoint
    from cropaggregation.evaluation import evaluate
    run=Path(a.run_dir);split='train' if a.split=='validation' else 'test'
    X,pids,y,meta=_load_common_split(str(ROOT/d['directory']),split)
    if (run/'normalization.npz').exists():
        with np.load(run/'normalization.npz') as z:low,high=z['minimum'],z['maximum']
    else:
        # Historical fixed-test checkpoints store their normalization as decimal CSV.
        low,high=read_minMaxVal(run/'min_Max.txt')
        # The archived NumPy 2.x training run calculated percentiles in float32.
        # Reading decimal CSV as float64 changes tree threshold decisions.
        low,high=low.astype(np.float32),high.astype(np.float32)
    if a.model=='STSMamba':
        X=extract_spatial_patches(str((ROOT/d['fixed_raster']).resolve()),meta,11,3);shape=(1,1,1,1,11)
        X=(X-low.reshape(shape))/(high.reshape(shape)-low.reshape(shape)+1e-8)
    else:X=(X.reshape(-1,7,11)-low)/(high-low+1e-8)
    X=np.clip(X,0,1).astype(np.float32);_set_seed(a.seed)
    if a.split=='validation':
        _,_,_,_,X,pids,y,_=extractValSet(X,pids,y,val_rate=.1,return_polygon_ids=True,sample_weights=np.ones(len(X),dtype=np.float32))
        meta=pd.concat([meta[meta.polygon_id.eq(pid)] for pid in pd.unique(pids)],ignore_index=True)
    assert np.array_equal(meta.polygon_id.to_numpy(),pids)
    model=get_model(a.model,11,3,7,a.device)
    load_model_checkpoint(model,run/'Best_model',a.device)
    temp=float((run/'calibration_temperature.txt').read_text())
    _,truth,_,prob,epistemic=evaluate(model,X,y,a.device,BATCH_SIZES,temperature=temp,mc_samples=1 if a.model=='XGBoost' else 30)
    result=meta[['polygon_id','ring_id','distance_px']].copy();result['y_true']=truth.cpu().numpy()
    result['epistemic']=epistemic.cpu().numpy()
    for i in range(3):result[f'prob_class_{i}']=prob.cpu().numpy()[:,i]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
    print('Saved',len(result),'pixels to',out)
if __name__=='__main__':main()
