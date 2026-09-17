"""Recompute archived experiment metrics using the same shared analysis implementation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def compare(actual, reference, keys, columns):
    a = actual.copy()
    b = reference.copy()
    for frame in (a, b):
        if 'seed' in frame:
            frame['seed'] = frame.seed.astype(str).str.replace('seed_', '', regex=False).astype(int)
    merged = a.merge(b, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError('Archived metric keys do not match reproduced metric keys')
    error = 0.0
    for col in columns:
        aa = merged[col+'_new'].to_numpy(float)
        bb = merged[col+'_old'].to_numpy(float)
        if not np.allclose(aa, bb, atol=1e-8, rtol=0, equal_nan=True):
            raise ValueError(f'Metric mismatch: {col}; maximum difference={np.nanmax(np.abs(aa-bb))}')
        error = max(error, float(np.nanmax(np.abs(aa-bb))))
    return {'rows': len(merged), 'max_absolute_error': error}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--section', choices=['fixed', 'spatial', 'temporal', 'all'], default='all')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/reproduction')
    p.add_argument('--compare-only', action='store_true', help='Compare already recomputed outputs without recalculating them')
    a = p.parse_args()
    sections = ['fixed', 'spatial', 'temporal'] if a.section == 'all' else [a.section]
    report = {}
    a.output.mkdir(parents=True, exist_ok=True)
    for section in sections:
        years = ['2021_2022', '2025_2026'] if section != 'temporal' else ['both']
        for year in years:
            dest = a.output/section/year
            base = ROOT/'results'/('fixed' if section == 'fixed' else 'generalization')
            command = [sys.executable, str(ROOT/'analyze.py'), '--experiment', section,
                       '--input', str(base), '--output', str(dest)]
            if year != 'both': command += ['--year', year]
            if not a.compare_only:
                subprocess.run(command, check=True, cwd=ROOT)
            if section == 'fixed':
                actual = pd.read_csv(dest/'fixed_metrics.csv').rename(columns={'d0':'selected_d0'})
                actual['scheme'] = actual.method.map({'M0':'M0_equal', 'M1':'M1_hard_px1', 'M2':'M2_train_calibrated_frozen'})
                reference = pd.read_csv(ROOT/'results/reference/fixed_test/run_metrics.csv')
                reference = reference[reference.year == year]
                checked = compare(actual, reference, ['model','seed','scheme'], ['brier','nll','macro_f1','oa','selected_d0'])
            else:
                actual = pd.read_csv(dest/'metrics.csv')
                if section == 'temporal':
                    actual = actual.rename(columns={'oa':'accuracy'})
                    actual['method'] = actual.method.map({'M0':'M0_equal','M2':'M2_source_frozen_one_se'})
                pattern = 'spatial_cv/'+year+'/*/seed_*/distance_selection/spatial_M0_M2_metrics.csv' if section=='spatial' else 'temporal/*/*/seed_*/temporal_M0_M2_metrics.csv'
                rows = []
                for file in base.glob(pattern):
                    frame = pd.read_csv(file)
                    run = file.parent.parent if section=='spatial' else file.parent
                    frame['model'] = run.parent.name
                    frame['seed'] = run.name
                    rows.append(frame)
                reference = pd.concat(rows, ignore_index=True)
                keys = ['model','seed','method'] + (['direction'] if section=='temporal' else [])
                checked = compare(actual, reference, keys, ['brier','nll','macro_f1','accuracy'])
            report[section+'/'+year] = checked
            print(section, year, checked, flush=True)
    (a.output/(a.section+'_verification.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
