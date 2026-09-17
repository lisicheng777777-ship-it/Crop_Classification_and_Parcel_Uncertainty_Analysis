"""Verify the exact small-file repository payload intended for ordinary Git upload."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='manifests/repository_sha256.csv'
EXCLUDED={'.git','.venv','__pycache__','.pytest_cache','outputs','release_assets','checkpoints','build','dist'}

def repository_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        if any(x in EXCLUDED or x.endswith('.egg-info') for x in rel.parts):continue
        if rel.parts[:2]==('data','raw'):continue
        if path.suffix in {'.pyc','.pyo','.lock','.part'} or path.name in {'.env','.DS_Store','Thumbs.db'}:continue
        if rel.as_posix()!=MANIFEST:yield path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write',action='store_true',help='Regenerate the checksum manifest after an intentional change')
    a=p.parse_args();files=list(repository_files());manifest=ROOT/MANIFEST
    rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    large=[r['path'] for r in rows if r['bytes']>=100*1024*1024]
    if large:raise ValueError('Files exceed GitHub ordinary-file limit: '+str(large))
    if a.write:
        manifest.parent.mkdir(exist_ok=True)
        with manifest.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
    with manifest.open(encoding='utf-8',newline='') as f:expected={r['path']:r for r in csv.DictReader(f)}
    actual={r['path']:r for r in rows}
    if set(actual)!=set(expected):raise ValueError('Repository file inventory changed; missing='+str(sorted(set(expected)-set(actual)))+'; extra='+str(sorted(set(actual)-set(expected))))
    for path,row in actual.items():
        if row['bytes']!=int(expected[path]['bytes']) or row['sha256']!=expected[path]['sha256']:raise ValueError('Checksum mismatch: '+path)
    print('Verified',len(rows),'repository files;',sum(r['bytes'] for r in rows),'bytes; no file reaches 100 MiB.')

if __name__=='__main__':main()
