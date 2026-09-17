import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('data_archive',Path(__file__).resolve().parents[1]/'scripts/data_archive.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)

class ArchiveTests(unittest.TestCase):
    def test_split_stream_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parts=root/'parts';parts.mkdir();(root/'manifests').mkdir()
            payload=bytes(range(256))*8
            writer=archive.PartsWriter(parts,128)
            with zipfile.ZipFile(writer,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('data/raw/example.bin',payload)
            writer.finish_part()
            self.assertGreater(len(writer.parts),1)
            (root/'manifests/data_release.json').write_text(json.dumps({'parts':writer.parts}))
            with (root/'manifests/data_sha256.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
                w.writerow({'path':'data/raw/example.bin','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
            previous=archive.ROOT;archive.ROOT=root
            try:
                archive.restore(parts)
                restored=root/'data/raw/example.bin'
                self.assertEqual(restored.read_bytes(),payload)
                restored.write_bytes(b'changed')
                with self.assertRaises(FileExistsError):archive.restore(parts)
                self.assertEqual(restored.read_bytes(),b'changed')
            finally:archive.ROOT=previous

    def test_corrupted_part_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parts=root/'parts';parts.mkdir();(root/'manifests').mkdir()
            p=parts/'crop-data.zip.001';p.write_bytes(b'wrong')
            (root/'manifests/data_release.json').write_text(json.dumps({'parts':[{'name':p.name,'bytes':5,'sha256':'0'*64}]}))
            previous=archive.ROOT;archive.ROOT=root
            try:
                with self.assertRaises(ValueError):archive.restore(parts,verify_only=True)
            finally:archive.ROOT=previous

if __name__=='__main__':unittest.main()
