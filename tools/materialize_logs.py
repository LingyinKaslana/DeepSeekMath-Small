"""Reassemble sanitized compressed logs from the snapshot manifest."""
import argparse,fnmatch,gzip,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pattern',required=True);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'EXPORT_MANIFEST.json').read_text());count=0
for record in manifest['files']:
 if record['kind']=='rollout_metrics' or not fnmatch.fnmatch(record['source'],a.pattern):continue
 parts=record['outputs']
 if not parts or not all(p['path'].endswith('.gz') for p in parts):continue
 target=root/parts[0]['path'].rsplit('.part',1)[0]
 assert root in target.resolve().parents
 if target.exists() and not a.overwrite:raise SystemExit(f'Already exists: {target.relative_to(root)}; use --overwrite')
 temporary=target.with_suffix(target.suffix+'.tmp')
 with temporary.open('wb') as writer:
  for part in parts:
   source=root/part['path'];assert root in source.resolve().parents
   with gzip.open(source,'rb') as reader:
    while block:=reader.read(1024*1024):writer.write(block)
 temporary.replace(target);count+=1;relative=str(target.relative_to(root))
 marker=root/'.local_materialized.json';local=json.loads(marker.read_text()) if marker.exists() else []
 if relative not in local:local.append(relative)
 marker.write_text(json.dumps(local,indent=2)+'\n')
 exclude=root/'.git/info/exclude'
 if exclude.parent.exists():
  current=exclude.read_text() if exclude.exists() else ''
  rule='/'+relative
  if rule not in current.splitlines():exclude.write_text(current+'\n'+rule+'\n')
 print(target.relative_to(root))
print(f'Materialized {count} log files. Contents remain the sanitized export, not original text generations.')
