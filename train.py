"""Train and predict fixed-test, spatial-CV or bidirectional temporal experiments."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['fixed','spatial','temporal'],required=True)
    p.add_argument('--config',default=str(ROOT/'configs/paper.json'))
    p.add_argument('--years',nargs='+',choices=['2025_2026','2021_2022'])
    p.add_argument('--models',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--epochs',type=int)
    p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--output',help='Dedicated output directory; do not mix different settings')
    p.add_argument('--dry-run',action='store_true',help='Validate inputs and print jobs without training')
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8'))
    years=a.years or (['2025_2026','2021_2022'] if a.experiment=='temporal' else ['2025_2026'])
    models=a.models or c['models'];seeds=a.seeds or c['seeds']
    if set(models)-set(c['models']):p.error('Unsupported models')
    if a.experiment=='temporal' and set(years)!={'2025_2026','2021_2022'}:p.error('Bidirectional temporal experiment requires both seasons')
    resolve=lambda s: str((ROOT/s).resolve())
    data_paths={y:resolve(c['datasets'][y]['directory']) for y in years}
    raster_key='fixed_raster' if a.experiment=='fixed' else 'generalization_raster'
    rasters={y:resolve(c['datasets'][y][raster_key]) for y in years}
    import numpy as np
    import pandas as pd
    for y,d in data_paths.items():
        for split in ['train','test']:
            f=pd.read_csv(Path(d)/f'samples_{split}_datasat.csv',header=None)
            m=pd.read_csv(Path(d)/f'samples_{split}_datasat_pixel_metadata.csv')
            assert f.shape[1]==79 and np.array_equal(f[1].to_numpy(int),m.polygon_id.to_numpy(int))
            assert set(f[0])<={1,2,3}
        if a.experiment!='fixed':
            folds=pd.read_csv(Path(d)/'spatial_folds.csv').rename(columns={'polygon_id':'parcel_id'})
            assert folds.groupby('spatial_group_30m').spatial_cv_fold.nunique().max()==1
            assert set(folds.spatial_cv_fold)==set(range(5))
        if 'STSMamba' in models and not Path(rasters[y]).is_file():raise FileNotFoundError('Configure raster: '+rasters[y])
    output=Path(a.output).resolve() if a.output else ROOT/c['output_root']/('fixed' if a.experiment=='fixed' else 'generalization')
    settings=dict(years=years,models=models,seeds=seeds,epochs=a.epochs or c['epochs'],device=a.device or c['device'],data=data_paths,rasters=rasters,output=str(output))
    print(json.dumps(dict(experiment=a.experiment,dry_run=a.dry_run,**settings),indent=2),flush=True)
    if a.dry_run:return
    output.mkdir(parents=True,exist_ok=True)
    # Protect resume from mixing incompatible settings (spatial and temporal share OOF caches).
    signature={k:settings[k] for k in ['epochs','device','data','rasters']}
    manifest=output/'training_settings.json'
    if manifest.exists():
        old=json.loads(manifest.read_text());
        for key in ['epochs','device']:
            if old[key]!=signature[key]:raise ValueError('Use a separate --output for changed '+key)
        for key in ['data','rasters']:
            for y,v in signature[key].items():
                if y in old[key] and old[key][y]!=v:raise ValueError('Input paths changed; use separate --output')
            signature[key]={**old[key],**signature[key]}
    manifest.write_text(json.dumps(signature,indent=2),encoding='utf-8')
    if a.experiment=='fixed':
        import torch
        from cropaggregation import fixed
        from cropaggregation.spatial_patches import extract_spatial_patches
        fixed.EPOCHS=settings['epochs']
        for y in years:
            train=fixed._load_common_split(data_paths[y],'train');test=fixed._load_common_split(data_paths[y],'test')
            patches={}
            if 'STSMamba' in models:
                patches={split:extract_spatial_patches(rasters[y],d[3],11,3) for split,d in [('train',train),('test',test)]}
            for model in models:
                for seed in seeds:
                    print('FIXED',y,model,seed,flush=True)
                    fixed._train_2021_checkpoint(model,'T0_standard',(1,1,1),train,test,patches,torch.device(settings['device']),output_root=str(output/'runs'/y/f'seed_{seed}'),seed=seed,source_label=y)
    else:
        from cropaggregation import generalization as g
        from cropaggregation.research import _load_dataset
        g.RASTERS=rasters
        args=SimpleNamespace(device=settings['device'],epoch=settings['epochs'],batch_size_list=[128,128,8292],monitor='kappa',learning_rate=3e-4,weight_decay=1e-3,class_weight='balanced',supplement_split_seed=c['split_seed'],input_dims='9,2',phenology_indices='4,5,6,7,8')
        data={};meta={};assign={};oof={};md={}
        for y in years:
            data[y]=_load_dataset(data_paths[y],11)
            meta[y]=pd.concat([g._metadata(data_paths[y],t) for t in ['train','test']],ignore_index=True)
            assign[y]=pd.read_csv(Path(data_paths[y])/'spatial_folds.csv').rename(columns={'polygon_id':'parcel_id'})
            assert set(np.unique(data[y]['all'][1])).issubset(set(assign[y].parcel_id))
            for model in models:
                md[y,model]=g._data_for_model(data[y],meta[y],y,model,output,11)
                for seed in seeds:
                    dest=output/'spatial_cv'/y/model/f'seed_{seed}'
                    cache=dest/'spatial_oof_pixel_predictions_with_distance.csv'
                    oof[y,model,seed]=pd.read_csv(cache) if cache.exists() else g._spatial_predictions(md[y,model],meta[y],assign[y],y,model,seed,args,output)
                    if a.experiment=='spatial':g._crossfit_distance(oof[y,model,seed],dest/'distance_selection')
        if a.experiment=='temporal':
            for source,target in [('2025_2026','2021_2022'),('2021_2022','2025_2026')]:
                for model in models:
                    for seed in seeds:g._temporal(md[source,model],md[target,model],meta[target],oof[source,model,seed],source,target,model,seed,args,output)
    print('COMPLETE',output,flush=True)

if __name__=='__main__':main()
