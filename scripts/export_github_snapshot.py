"""Export a read-only, bounded snapshot of code, numerical logs and provenance.

No weights, dataset bodies, environments or caches are copied. Large JSONL files
are compressed in independently valid parts; rollout records retain numerical
statistics and identities, with source text/token arrays omitted. Run while the
source training appends logs: readers consume only their initial byte boundary.
"""
import argparse,csv,io,gzip,hashlib,json,os,re,shutil,time
from pathlib import Path
from datetime import datetime,timezone

SECRET=re.compile(r'(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})')
BODIES={'question','prompt','completion','raw_completion','solution','response','text','problem','generated_text','reference_answer'}
TOKEN_ARRAYS={'tokens','prompt_ids','input_ids','labels','attention_mask'}
SKIP_DIRS={'vendor','.git','__pycache__','.cache','.venv','.venv-plot','node_modules'}
SKIP_SUFFIX={'.pt','.pth','.bin','.safetensors','.npy','.npz','.parquet','.arrow','.pyc','.html','.so','.o'}
TEXT_SUFFIX={'.py','.sh','.json','.jsonl','.log','.txt','.md','.csv','.yaml','.yml','.toml','.svg','.gitignore'}
CHUNK=16*1024*1024


def sha(data):return hashlib.sha256(data).hexdigest()
def clean(text,root):
    text=text.replace(str(root),'.')
    # Do not export source-machine user/home paths. External assets remain explicit.
    home=str(root.parent.parent)
    if home.startswith('/localhome/'):
        text=text.replace(home+'/lingyin/nvidia-perfsdk','/external/nvidia-perfsdk')
        text=text.replace(home,'/external/home')
        text=text.replace(Path(home).name,'LOCAL_USER')
    text=text.replace('GB110','LOCAL_GPU_HOST')
    return SECRET.sub('REDACTED_CREDENTIAL',text)


def numeric_record(value):
    if isinstance(value,list):return [numeric_record(x) for x in value]
    if not isinstance(value,dict):return value
    sample=bool({'completion','raw_completion','question','prompt_ids','input_ids'}&value.keys())
    result={}
    for key,item in value.items():
        if sample and key in BODIES and isinstance(item,str):
            result[key+'_sha256']=sha(item.encode());result[key+'_characters']=len(item)
        elif sample and key in TOKEN_ARRAYS and isinstance(item,list):
            result[key+'_count']=len(item)
        else:result[key]=numeric_record(item)
    return result


def group_metrics(record):
    samples=record['samples'];first=samples[0] if samples else {}
    result={k:v for k,v in record.items() if k!='samples'}
    correct=[bool(r['correct']) for r in samples]
    result.update(source_id=first.get('source_id'),question_sha256=first.get('question_sha256'),
                  samples=len(samples),correct=correct,correct_count=sum(correct),
                  response_tokens=[len(r['tokens']) for r in samples],
                  prompt_tokens=[len(r.get('prompt_ids',[])) for r in samples],
                  finish_reasons=[r.get('finish_reason') for r in samples],
                  rewards=[r.get('reward') for r in samples],
                  raw_correct=[r.get('raw_correct',r['correct']) for r in samples],
                  forced_incorrect=[r.get('forced_incorrect',False) for r in samples])
    if any(r.get('cancellation') for r in samples):result['cancellations']=[r.get('cancellation') for r in samples]
    return result


class Parts:
    def __init__(self,destination,compressed):
        self.destination=destination;self.compressed=compressed;self.stream=None;self.size=0;self.paths=[]
    def write(self,data):
        if self.stream is None or (self.compressed and self.size>=CHUNK):
            if self.stream:self.stream.close()
            path=self.destination.with_name(self.destination.name+f'.part{len(self.paths)+1:04d}.gz') if self.compressed else self.destination
            path.parent.mkdir(parents=True,exist_ok=True);self.paths.append(path);self.size=0
            self.stream=gzip.GzipFile(filename=str(path),mode='wb',compresslevel=6,mtime=0) if self.compressed else path.open('wb')
        self.stream.write(data);self.size+=len(data)
    def close(self):
        if self.stream:self.stream.close()


def enumerate_files(root):
    files=[];excluded=[]
    for top in ('scripts','configs','runs'):
        for directory,dirs,names in os.walk(root/top):
            keep=[]
            for name in dirs:
                path=Path(directory)/name
                if name in SKIP_DIRS:excluded.append(str(path.relative_to(root)));continue
                if '.incomplete' in name or '.previous' in name:
                    excluded.append(str(path.relative_to(root)));continue
                if 'checkpoint' in name:
                    # Only JSON manifests are retained from weight directories.
                    for manifest in path.rglob('checkpoint_manifest.json'):
                        rel=manifest.relative_to(root);base=Path(directory).relative_to(root)
                        suffix=manifest.parent.relative_to(Path(directory))
                        files.append((manifest,base/'checkpoint_metadata'/Path(str(suffix)+'.json')))
                    excluded.append(str(path.relative_to(root)));continue
                keep.append(name)
            dirs[:]=keep
            for name in names:
                path=Path(directory)/name;rel=path.relative_to(root)
                if path.suffix in SKIP_SUFFIX or name.endswith('.tmp') or name.startswith('.env') or name.endswith('_paper.pdf'):
                    excluded.append(str(rel));continue
                if path.suffix in TEXT_SUFFIX or path.suffix in {'.png','.pdf','.jpg','.jpeg'}:
                    files.append((path,rel))
    for path in root.glob('*.log'):files.append((path,Path('logs')/path.name))
    for name in ('activate.sh','requirements.txt'):
        if (root/name).exists():files.append((root/name,Path(name)))
    for path in (root/'data').rglob('manifest.json'):
        if '.cache' not in path.parts:files.append((path,path.relative_to(root)))
    return files,excluded


def export(root,output):
    root=root.resolve();output=output.resolve();assert root!=output and root not in output.parents
    output.mkdir(parents=True,exist_ok=False)
    files,excluded=enumerate_files(root);records=[];started=datetime.now(timezone.utc).isoformat()
    for index,(source,relative) in enumerate(files):
        initial=source.stat();kind='sanitized_text';digest=hashlib.sha256();rows=0;dropped=0;cutoff=None
        is_group='rollouts' in source.name and source.suffix=='.jsonl'
        if source.name=='before_transition_rollouts.jsonl':
            path=source.parent/'boundary.json'
            if path.exists():cutoff=json.loads(path.read_text())['completed_rounds']
        if source.name=='before_recovery_rollouts.jsonl':
            path=source.parent/'recovery.json'
            if path.exists():cutoff=json.loads(path.read_text())['resume_round']
        if is_group:relative=relative.with_name(relative.name.replace('rollouts','rollout_metrics'))
        destination=output/relative
        if source.suffix=='.jsonl':
            writer=Parts(destination,initial.st_size>8*1024*1024 or is_group)
            consumed=0
            with source.open('rb') as stream:
                boundary=os.fstat(stream.fileno()).st_size
                while consumed<boundary:
                    line=stream.readline(boundary-consumed);consumed+=len(line);digest.update(line)
                    if not line.strip():continue
                    try:item=json.loads(line)
                    except json.JSONDecodeError:
                        if consumed==boundary and not line.endswith(b'\n'):dropped+=1;continue
                        raise
                    if cutoff is not None and item.get('round',0)<=cutoff:dropped+=1;continue
                    if is_group and isinstance(item,dict) and isinstance(item.get('samples'),list):item=group_metrics(item)
                    else:item=numeric_record(item)
                    data=clean(json.dumps(item,ensure_ascii=False,separators=(',',':')),root).encode()+b'\n'
                    writer.write(data);rows+=1
            writer.close();paths=writer.paths;kind='rollout_metrics' if is_group else 'jsonl_metrics'
        else:
            with source.open('rb') as stream:
                boundary=os.fstat(stream.fileno()).st_size;data=stream.read(boundary)
            digest.update(data)
            if source.suffix in {'.png','.pdf','.jpg','.jpeg'}:
                if b'/localhome/' in data or SECRET.search(data.decode('latin1')):
                    excluded.append(str(source.relative_to(root)));continue
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);paths=[destination];kind='generated_figure'
            else:
                text=data.decode('utf-8')
                if source.suffix=='.json':
                    try:obj=json.loads(text)
                    except json.JSONDecodeError:raise
                    # Structured summaries are kept; example bodies are removed recursively.
                    text=json.dumps(numeric_record(obj),ensure_ascii=False,indent=2)+'\n'
                if source.suffix=='.csv':
                    reader=csv.DictReader(io.StringIO(text))
                    bodies=[c for c in (reader.fieldnames or []) if c=='question' or c.endswith('_completion') or c in ('prompt','completion','raw_completion')]
                    if bodies:
                        fields=[c for c in reader.fieldnames if c not in bodies]
                        for c in bodies:fields.extend((c+'_sha256',c+'_characters'))
                        buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
                        for row in reader:
                            for c in bodies:
                                value=row.pop(c);row[c+'_sha256']=sha(value.encode());row[c+'_characters']=len(value)
                            writer.writerow(row)
                        text=buffer.getvalue();kind='csv_metrics'
                data=clean(text,root).encode()
                writer=Parts(destination,len(data)>8*1024*1024)
                for start in range(0,len(data),CHUNK):writer.write(data[start:start+CHUNK])
                if not data:writer.write(b'')
                writer.close();paths=writer.paths
                if source.suffix=='.sh':
                    for path in paths:path.chmod(0o755)
        records.append({'source':str(source.relative_to(root)),'source_bytes':boundary,'source_snapshot_sha256':digest.hexdigest(),
                        'kind':kind,'records':rows,'omitted_records':dropped,'archived_branch_after_round':cutoff,
                        'outputs':[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in paths]})
        if index%100==0:print(json.dumps({'files_done':index,'files_total':len(files),'source':str(source.relative_to(root))}),flush=True)
    manifest={'schema_version':1,'snapshot_started_utc':started,'snapshot_finished_utc':datetime.now(timezone.utc).isoformat(),
              'source_project':'DeepSeekMath-Small','files':records,'excluded_paths':excluded,
              'policies':['Source training is untouched; append-only files bounded at initial file size.',
                          'Weights, tensor dumps, environment caches, source datasets and third-party vendor excluded.',
                          'Example bodies/token IDs replaced by hashes/counts; correctness/rewards/timing retained.',
                          'Large files compressed as concatenable gzip parts; original/export hashes distinguish path sanitization.',
                          'Archived failed-branch rollout metrics retain only records after their resume boundary.']}
    (output/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'complete','destination':str(output),'files':len(records),'output_bytes':sum(x['bytes'] for r in records for x in r['outputs'])}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.source,a.output)
