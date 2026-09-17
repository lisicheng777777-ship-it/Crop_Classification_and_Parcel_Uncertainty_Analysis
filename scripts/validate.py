"""Validate publication scope, data alignment, code duplication and optional checkpoint compatibility."""
import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/validation.json')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/paper.json').read_text())
    hashes=defaultdict(list);python_files=[]
    for path in ROOT.rglob('*.py'):
        if any(p in {'outputs','.venv','__pycache__'} for p in path.relative_to(ROOT).parts):continue
        raw=path.read_bytes();ast.parse(raw,filename=str(path));python_files.append(path)
        if raw.strip():hashes[hashlib.sha256(raw).hexdigest()].append(path.relative_to(ROOT).as_posix())
    duplicates=[paths for paths in hashes.values() if len(paths)>1]
    if duplicates:raise ValueError('Duplicate source files: '+str(duplicates))
    datasets={}
    for year,settings in config['datasets'].items():
        folder=ROOT/settings['directory'];ids={};counts={}
        for split in ['train','test']:
            frame=pd.read_csv(folder/f'samples_{split}_datasat.csv',header=None)
            meta=pd.read_csv(folder/f'samples_{split}_datasat_pixel_metadata.csv')
            assert frame.shape[1]==79 and len(frame)==len(meta)
            assert np.isfinite(frame.to_numpy(float)).all()
            assert set(frame[0])=={1,2,3}
            assert np.array_equal(frame[1].to_numpy(int),meta.polygon_id.to_numpy(int))
            assert meta.distance_px.ge(0).all()
            assert frame.groupby(1)[0].nunique().max()==1
            ids[split]=set(frame[1].astype(int));counts[split]={'pixels':len(frame),'parcels':len(ids[split])}
        assert ids['train'].isdisjoint(ids['test'])
        folds=pd.read_csv(folder/'spatial_folds.csv').rename(columns={'parcel_id':'polygon_id'})
        assert set(folds.polygon_id)==ids['train']|ids['test']
        assert folds.polygon_id.is_unique
        assert set(folds.spatial_cv_fold)==set(range(5))
        assert folds.groupby('spatial_group_30m').spatial_cv_fold.nunique().max()==1
        datasets[year]=counts
    matrix=[]
    for year in config['datasets']:
        for model in config['models']:
            for seed in config['seeds']:
                fixed=ROOT/'results/fixed/runs'/year/f'seed_{seed}'/'T0_standard'/model
                for name in ['source_test_pixel_probabilities.csv','source_validation_pixel_probabilities.csv']:
                    path=fixed/name;assert path.is_file(),path
                    frame=pd.read_csv(path);cols=[f'prob_class_{i}' for i in range(3)]
                    probs=frame[cols].to_numpy(float)
                    assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
                spatial=ROOT/'results/generalization/spatial_cv'/year/model/f'seed_{seed}'
                assert (spatial/'spatial_oof_pixel_predictions_with_distance.csv').is_file()
                for fold in range(5):assert (spatial/f'fold_{fold}'/'pixel_predictions.csv').is_file()
                matrix.append({'year':year,'model':model,'seed':seed})
    for direction in ['2021_2022_to_2025_2026','2025_2026_to_2021_2022']:
        for model in config['models']:
            for seed in config['seeds']:
                run=ROOT/'results/generalization/temporal'/direction/model/f'seed_{seed}'
                assert (run/'target_pixel_predictions_with_distance.csv').is_file()
                assert (run/'frozen_source_d0.csv').is_file()
    checkpoint_count=None
    if args.checkpoints:
        import torch
        from cropaggregation.modeling import get_model,load_model_checkpoint
        torch.set_num_threads(2)
        checkpoint_count=0
        for model_name in config['models']:
            model=get_model(model_name,11,3,7,'cpu')
            paths=[p for p in (ROOT/'checkpoints').rglob('Best_model') if model_name in p.parts]
            assert len(paths)==70,(model_name,len(paths))
            for path in paths:
                load_model_checkpoint(model,path,'cpu');model.eval()
                sample=torch.zeros((2,7,3,3,11) if model_name=='STSMamba' else (2,7,11))
                with torch.no_grad():prediction=model(sample)
                assert tuple(prediction.shape)==(2,3) and torch.isfinite(prediction).all()
                checkpoint_count+=1
            print('Loaded and checked',model_name,len(paths),'checkpoints',flush=True)
    report={'python_files':len(python_files),'duplicate_code_groups':duplicates,'datasets':datasets,'fixed_configurations':len(matrix),'spatial_configurations':len(matrix),'spatial_training_folds':len(matrix)*5,'temporal_configurations':50,'checkpoint_forward_checks':checkpoint_count,'scope':'syntax, content duplicate scan, data alignment, split structure, experiment matrix and optional checkpoint load/forward'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
