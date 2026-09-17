"""Build, download, verify and restore the raw-data/checkpoint release attachment."""
import argparse
import bisect
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CHUNK=8*1024*1024

def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()

class PartsWriter(io.RawIOBase):
    def __init__(self,directory,limit):
        self.directory=directory;self.limit=limit;self.position=0;self.parts=[];self.file=None;self.part_size=0
    def writable(self):return True
    def tell(self):return self.position
    def seekable(self):return False
    def finish_part(self):
        if self.file is not None:
            self.file.close()
            self.parts.append({'name':self.path.name,'bytes':self.part_size,'sha256':self.hash.hexdigest(),'url':None})
            self.file=None
    def write(self,data):
        total=len(data);view=memoryview(data)
        while len(view):
            if self.file is None:
                self.path=self.directory/f'crop-data.zip.{len(self.parts)+1:03d}'
                self.file=self.path.open('xb');self.part_size=0;self.hash=hashlib.sha256()
            n=min(len(view),self.limit-self.part_size)
            block=view[:n];self.file.write(block);self.hash.update(block)
            self.part_size+=n;self.position+=n;view=view[n:]
            if self.part_size==self.limit:self.finish_part()
        return total

class PartsReader(io.RawIOBase):
    def __init__(self,paths):
        self.files=[p.open('rb') for p in paths];self.ends=[];total=0
        for p in paths:total+=p.stat().st_size;self.ends.append(total)
        self.size=total;self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        new=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if new<0:raise ValueError('Negative seek')
        self.position=new;return new
    def read(self,n=-1):
        remaining=self.size-self.position if n<0 else min(n,self.size-self.position)
        blocks=[]
        while remaining>0:
            i=bisect.bisect_right(self.ends,self.position)
            start=0 if i==0 else self.ends[i-1]
            count=min(remaining,self.ends[i]-self.position)
            f=self.files[i];f.seek(self.position-start);block=f.read(count)
            if len(block)!=count:raise IOError('Truncated archive part')
            blocks.append(block);self.position+=count;remaining-=count
        return b''.join(blocks)
    def close(self):
        for f in self.files:f.close()
        super().close()

def build(directory):
    directory.mkdir(parents=True,exist_ok=True)
    if list(directory.glob('crop-data.zip.*')):raise FileExistsError('Use an empty release directory')
    files=sorted(p for folder in ['data/raw','checkpoints'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix!='.lock')
    rows=[];sink=PartsWriter(directory,1900*1024*1024)
    with zipfile.ZipFile(sink,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for index,path in enumerate(files):
            name=path.relative_to(ROOT).as_posix();before=path.stat();h=hashlib.sha256()
            info=zipfile.ZipInfo.from_file(path,name);info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1
            with path.open('rb') as src,archive.open(info,'w',force_zip64=True) as dst:
                for block in iter(lambda:src.read(CHUNK),b''):dst.write(block);h.update(block)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during archive: '+name)
            rows.append({'path':name,'bytes':after.st_size,'sha256':h.hexdigest()})
            if index%50==0 or after.st_size>100*1024*1024:print('Archived',index+1,'/',len(files),name,flush=True)
    sink.finish_part()
    manifests=ROOT/'manifests';manifests.mkdir(exist_ok=True)
    with (manifests/'data_sha256.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
    metadata={'format':'concatenated-zip64-parts','files':len(rows),'uncompressed_bytes':sum(r['bytes'] for r in rows),'parts':sink.parts,'published':False}
    (manifests/'data_release.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (directory/'data_release.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Built data release:',len(sink.parts),'parts;',len(rows),'files',flush=True)

def restore(directory,verify_only=False,download=False):
    metadata=json.loads((ROOT/'manifests/data_release.json').read_text(encoding='utf-8'))
    directory.mkdir(parents=True,exist_ok=True);paths=[]
    for part in metadata['parts']:
        path=directory/part['name']
        if not path.exists() and download:
            if not part.get('url'):raise ValueError('Release has not been published; supply the release parts locally or configure their real URLs')
            temp=path.with_suffix(path.suffix+'.part')
            urllib.request.urlretrieve(part['url'],temp)
            if temp.stat().st_size!=part['bytes'] or digest_file(temp)!=part['sha256']:raise ValueError('Downloaded part checksum mismatch')
            temp.replace(path)
        if not path.exists() or path.stat().st_size!=part['bytes'] or digest_file(path)!=part['sha256']:raise ValueError('Missing or corrupt archive part: '+str(path))
        paths.append(path)
        print('Verified archive part:',part['name'],flush=True)
    with (ROOT/'manifests/data_sha256.csv').open(encoding='utf-8',newline='') as f:rows={row['path']:row for row in csv.DictReader(f)}
    with PartsReader(paths) as reader,zipfile.ZipFile(reader) as archive:
        names=[i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names)!=len(set(names)) or set(names)!=set(rows):raise ValueError('Archive members differ from manifest')
        for index,name in enumerate(names):
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not name.startswith(('data/raw/','checkpoints/')):raise ValueError('Unsafe member: '+name)
            record=rows[name];h=hashlib.sha256();size=0;dst=None;temp=None
            if not verify_only:
                if path.exists():
                    if path.stat().st_size!=int(record['bytes']) or digest_file(path)!=record['sha256']:raise FileExistsError('Refusing to overwrite a different local file: '+str(path))
                else:
                    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.part');dst=temp.open('xb')
            try:
                with archive.open(name) as src:
                    for block in iter(lambda:src.read(CHUNK),b''):
                        h.update(block);size+=len(block)
                        if dst:dst.write(block)
            finally:
                if dst:dst.close()
            if size!=int(record['bytes']) or h.hexdigest()!=record['sha256']:raise ValueError('Content checksum mismatch: '+name)
            if temp:temp.replace(path)
            if index%100==0:print('Verified',index+1,'/',len(names),flush=True)
    print('Data archive verified:',len(rows),'files',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','restore','verify'])
    p.add_argument('--directory',type=Path,default=ROOT/'release_assets')
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    if a.action=='build':build(a.directory)
    else:restore(a.directory,a.action=='verify',a.download)

if __name__=='__main__':main()
