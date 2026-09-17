"""Check raster metadata and selected pixel centres against distributed feature tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'configs/paper.json').read_text());reports=[]
    for year,d in cfg['datasets'].items():
        for kind in ['fixed_raster','generalization_raster']:
            with rasterio.open(ROOT/d[kind]) as src:
                assert src.count==77
                for split in ['train','test']:
                    f=pd.read_csv(ROOT/d['directory']/f'samples_{split}_datasat.csv',header=None)
                    m=pd.read_csv(ROOT/d['directory']/f'samples_{split}_datasat_pixel_metadata.csv')
                    ids=np.linspace(0,len(m)-1,min(12,len(m)),dtype=int)
                    differences=[]
                    for i in ids:
                        row,col=src.index(float(m.iloc[i].x),float(m.iloc[i].y))
                        assert 0<=row<src.height and 0<=col<src.width
                        values=src.read(window=Window(col,row,1,1))[:,0,0]
                        expected=f.iloc[i,2:].to_numpy(float)
                        differences.append(float(np.max(np.abs(values-expected))))
                        if not np.allclose(values,expected,rtol=1e-5,atol=1e-5,equal_nan=True):
                            raise ValueError(f'Raster/features differ: {year} {kind} {split} row={i}, max={differences[-1]}')
                    reports.append({'year':year,'raster':d[kind],'split':split,'sampled_pixels':len(ids),'max_absolute_difference':max(differences),'bands':src.count,'crs':str(src.crs)})
    dest=ROOT/'outputs/raw_validation.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('Raster/feature checks passed:',sum(r['sampled_pixels'] for r in reports),'sampled pixels')

if __name__=='__main__':main()
