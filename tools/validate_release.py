"""Check release file limits, credentials/paths, JSONL counts and SHA256 integrity."""
import argparse,gzip,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
SECRET=re.compile(rb'(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})')
PRIVATE_PATH=re.compile(rb'/localhome/[A-Za-z0-9_.-]+/')
PRIVATE_KEY=b'-----BEGIN '+b'PRIVATE KEY-----'
FORBIDDEN={'.pt','.pth','.bin','.safetensors','.npy','.npz','.parquet','.arrow','.pyc'}
OMIT={'RELEASE_MANIFEST.json','PUBLICATION_CHECKS.json'}
def files():
 local=set()
 marker=ROOT/'.local_materialized.json'
 if marker.exists():
  manifest=json.loads((ROOT/'EXPORT_MANIFEST.json').read_text())
  allowed={r['outputs'][0]['path'].rsplit('.part',1)[0] for r in manifest['files'] if r['outputs'] and all(o['path'].endswith('.gz') for o in r['outputs']) and r['kind']!='rollout_metrics'}
  local=set(json.loads(marker.read_text()));assert local<=allowed,'Invalid local materialization marker'
 return sorted(p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.name not in OMIT and p.name!='.local_materialized.json' and str(p.relative_to(ROOT)) not in local)
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def validate(write=False):
 errors=[];paths=files();total=0
 for path in paths:
  relative=str(path.relative_to(ROOT));size=path.stat().st_size;total+=size
  if size>=20*1024*1024:errors.append((relative,'file >=20MiB'))
  if path.suffix in FORBIDDEN:errors.append((relative,'model/data/binary asset'))
  if path.suffix in {'.png','.pdf','.jpg','.jpeg'}:continue
  stream=gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')
  with stream:
   tail=b''
   while chunk:=stream.read(1024*1024):
    chunk=tail+chunk
    if SECRET.search(chunk) or PRIVATE_PATH.search(chunk) or PRIVATE_KEY in chunk:
     errors.append((relative,'credential or source-home path'));break
    tail=chunk[-256:]
  if path.suffix=='.json':
   try:json.loads(path.read_text())
   except Exception as exc:errors.append((relative,'JSON parse: '+str(exc)))
 export=json.loads((ROOT/'EXPORT_MANIFEST.json').read_text());jsonl_rows=0;rollout_groups=0
 for record in export['files']:
  for output in record['outputs']:
   path=ROOT/output['path']
   if not path.exists():errors.append((output['path'],'missing export output'));continue
   if path.stat().st_size!=output['bytes'] or digest(path)!=output['sha256']:errors.append((output['path'],'export hash mismatch'))
  if record['kind'] in ('jsonl_metrics','rollout_metrics'):
   count=0
   for output in record['outputs']:
    path=ROOT/output['path'];opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as stream:
     for line in stream:
      if line.strip():json.loads(line);count+=1
   if count!=record['records']:errors.append((record['source'],'JSONL record count mismatch'))
   jsonl_rows+=count
   if record['kind']=='rollout_metrics':rollout_groups+=count
 release=ROOT/'RELEASE_MANIFEST.json'
 entries=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]
 if write:
  release.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'files':entries},indent=2)+'\n')
 elif release.exists():
  expected={e['path']:e for e in json.loads(release.read_text())['files']};actual={e['path']:e for e in entries}
  if expected!=actual:errors.append(('RELEASE_MANIFEST.json','release file set or hashes changed'))
 report={'status':'passed' if not errors else 'failed','files':len(paths),'bytes':total,
         'largest_file_bytes':max(p.stat().st_size for p in paths),'validated_jsonl_records':jsonl_rows,'rollout_metric_records':rollout_groups,
         'errors':errors,'checked_utc':datetime.now(timezone.utc).isoformat()}
 (ROOT/'PUBLICATION_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 if errors:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-manifest',action='store_true');a=p.parse_args();validate(a.write_manifest)
