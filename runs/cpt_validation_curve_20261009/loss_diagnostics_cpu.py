"""CPU-only descriptive loss diagnostics; no causal or native-equivalence claims."""
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
from urllib.parse import urlparse
import numpy as np

root=Path('.')
orig=root/'runs/cpt_10b_20261008'
resume=root/'runs/cpt_10b_20261009_resumed'
out=root/'runs/cpt_validation_curve_20261009'
manifest=json.loads((orig/'manifest.json').read_text())
data=Path(manifest['arguments']['data'])
data_meta=json.loads((orig/'data_manifest.json').read_text())
schedule=manifest['schedule']
points=json.loads((orig/'validation_history.json').read_text())
if (resume/'validation_history.json').exists():
    for p in json.loads((resume/'validation_history.json').read_text()):
        if p['step']>175004:points.append(p)
point_steps={p['step'] for p in points}

def summary(rows):
    losses=np.asarray([r['loss'] for r in rows],dtype=np.float64)
    grads=np.asarray([r['grad_norm'] for r in rows],dtype=np.float64)
    return {'updates':len(rows),'first_step':rows[0]['step'],'last_step':rows[-1]['step'],
       'train_loss_mean':float(losses.mean()),'train_loss_sample_sd':float(losses.std(ddof=1)),
       'train_loss_nominal_standard_error':float(losses.std(ddof=1)/math.sqrt(len(rows))),
       'grad_norm_mean':float(grads.mean()),'grad_norm_median':float(np.median(grads)),
       'grad_norm_p95':float(np.percentile(grads,95)),'grad_norm_max':float(grads.max()),
       'grad_norm_gt_1_fraction':float((grads>1).mean()),'grad_norm_gt_10_fraction':float((grads>10).mean()),
       'grad_norm_gt_100_fraction':float((grads>100).mean()),
       'nonfinite_loss_count':int((~np.isfinite(losses)).sum()),'nonfinite_grad_norm_count':int((~np.isfinite(grads)).sum()),
       'native_retry_updates':sum(bool(r.get('native_retry_applied')) for r in rows),
       'lr_first':rows[0]['lr'],'lr_last':rows[-1]['lr'],
       'input_tokens':sum(r['input_tokens'] for r in rows)}

windows={};segment_rows=collections.defaultdict(list);epoch_stats=collections.defaultdict(lambda:{'loss':[],'grad':[]})
recent=collections.deque(maxlen=500)
segments={'first_500':(1,500),'warmup_end_500':(14760,15259),
          'epoch1_end_500':(152089,152588),'epoch2_first_500':(152589,153088),
          'last_500_before_original_failure':(175497,175996)}
for run in [orig,resume]:
    if not (run/'training.jsonl').exists():continue
    with (run/'training.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # possible concurrent partial final write
            if run==orig:
                recent.append(r)
                epoch_stats[r['epoch']]['loss'].append(r['loss']);epoch_stats[r['epoch']]['grad'].append(r['grad_norm'])
                for name,(start,end) in segments.items():
                    if start<=r['step']<=end:segment_rows[name].append(r)
                if r['step'] in point_steps:
                    windows[r['step']]={str(n):summary(list(recent)[-n:]) for n in (100,500)}
            elif r['step']>175004:
                recent.append(r)
                if r['step'] in point_steps:
                    windows[r['step']]={str(n):summary(list(recent)[-n:]) for n in (100,500)}
    if run==orig:orig_last=r
    else:resume_last=r
validation_rows=[]
for p in sorted(points,key=lambda p:p['step']):
    w=windows.get(p['step'])
    if w:
        for values in w.values():values['native_val_minus_compiled_train_loss']=p['loss']-values['train_loss_mean']
    validation_rows.append({'step':p['step'],'native_validation_loss':p['loss'],'windows':w})
epochs=[]
for epoch,x in epoch_stats.items():
    losses=np.asarray(x['loss']);grads=np.asarray(x['grad'])
    epochs.append({'epoch':epoch,'updates':len(losses),'train_loss_mean':float(losses.mean()),
      'grad_norm_mean':float(grads.mean()),'grad_norm_median':float(np.median(grads)),
      'grad_norm_max':float(grads.max()),'grad_norm_gt_10_fraction':float((grads>10).mean()),
      'grad_norm_gt_100_fraction':float((grads>100).mean()),'nonfinite_grad_norm_count':int((~np.isfinite(grads)).sum())})

val_documents=[]
with (data/'validation.jsonl').open() as f:
    for line in f:
        x=json.loads(line);x.pop('text',None);val_documents.append(x)
meta_path=data/'train.jsonl';meta_size=meta_path.stat().st_size
def doc_at(target):
    with meta_path.open('rb') as f:
        lo,hi=0,meta_size
        while hi-lo>8192:
            mid=(lo+hi)//2;f.seek(mid)
            if mid:f.readline()
            pos=f.tell();line=f.readline()
            if not line:hi=mid;continue
            r=json.loads(line)
            if r['token_offset']<=target:lo=pos
            else:hi=mid
        f.seek(lo)
        for line in f:
            r=json.loads(line)
            if r['token_offset']<=target<r['token_offset']+r['used_tokens']:return r
    raise RuntimeError('No metadata at requested token offset')
sample=[];offset=0
for shard in data_meta['shard_sampling']:
    count=shard['tokens']['train']
    for fraction in (.1,.5,.9):
        r=doc_at(offset+int(count*fraction));assert r['source_shard']==shard['shard']
        sample.append(r)
    offset+=count
assert offset==5000000000

def document_summary(rows):
    used=np.asarray([r['used_tokens'] for r in rows]);full=np.asarray([r['document_tokens_with_eos'] for r in rows])
    domains=collections.Counter(urlparse(r['url']).netloc for r in rows)
    domain_tokens=collections.Counter()
    for r in rows:domain_tokens[urlparse(r['url']).netloc]+=r['used_tokens']
    return {'documents':len(rows),'used_tokens_mean':float(used.mean()),'used_tokens_median':float(np.median(used)),
      'document_tokens_before_quota_truncation_mean':float(full.mean()),'document_tokens_before_quota_truncation_median':float(np.median(full)),
      'prefix_truncated_documents':sum(r['prefix_truncated'] for r in rows),
      'source_shards':len({r['source_shard'] for r in rows}),'url_domains':len(domains),
      'top_domains_by_document_count':domains.most_common(10),
      'top_domains_by_summed_document_used_tokens':domain_tokens.most_common(10)}
pilot=[r for r in val_documents if r['token_offset']<262144]
new=[r for r in val_documents if r['token_offset']>=262144]
val_hash={r['text_sha256'] for r in val_documents};val_urls={r['url'] for r in val_documents}
summary_data={'manifest':{k:data_meta[k] for k in ['splits','sampling','split','packing','train_validation_text_hash_overlap','train_validation_url_overlap','exact_global_selected_text_and_url_deduplication','rejected_global_text_or_url_duplicates']},
    'validation_all':document_summary(val_documents),'validation_preserved_pilot':document_summary(pilot),
    'validation_new_extension':document_summary(new),
    'train_metadata_sample':document_summary(sample),'train_sampling':'three token-offset probes at 10%,50%,90% of each shard quota (342 documents); not a uniform-document or complete domain census',
    'sample_train_validation_text_hash_overlap':len({r['text_sha256'] for r in sample}&val_hash),
    'sample_train_validation_url_overlap':len({r['url'] for r in sample}&val_urls),
    'selected_train_doc_mean_used_tokens_full_dataset':5000000000/2540257,
    'token_file_sha256_recorded':manifest['signature']['token_file_sha256'],
    'source_sampling_note':'Both splits cover the same 114 source shards with nearly equal token quotas; validation includes 262144 preserved pilot tokens and quota-truncated prefixes. Domain proportions cannot be established from the 342-document train sample.'}
diagnostics={'status':'completed','cpu_only':True,'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'schedule':schedule,'epoch_steps':152588,'epoch_boundary_input_tokens':5000000000,
    'warmup_end_input_tokens':15259*32768,
    'lr_decay_1_based_updates':{'80_percent':244142,'90_percent':274660},
    'input_tokens_completed_before_lr_decay':{'80_percent':244141*32768-5120,'90_percent':274659*32768-5120},
    'validation_and_preceding_training_windows':validation_rows,
    'selected_training_segments':{name:summary(rows) for name,rows in segment_rows.items()},
    'original_logged_successful_updates_by_epoch':epochs,
    'last_original_logged_update':orig_last,'last_resumed_logged_update':resume_last,
    'data_comparison':summary_data,
    'interpretation_limits':['Training losses use the compiled training forward; validation uses native eager. Their arithmetic difference is descriptive, not a calibrated generalization gap.',
       'Standard errors assume independent updates and ignore shuffled-block dependence; they are nominal summaries, not formal significance tests.',
       'Only successful-update gradient norms are logged. The failure step does not have a saved norm/loss, so nonfinite rates among successful updates exclude that failure.',
       'Warmup, elapsed training exposure and validation loss move together; these observations do not establish learning-rate causality.',
       'Epoch two reuses the same 5B token blocks in a fresh permutation. Epoch differences combine parameter changes, learning-rate stage and batch composition.']}
out.mkdir(parents=True,exist_ok=True)
(out/'loss_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(out/'loss_diagnostics.json'),'windows':validation_rows[::5]+validation_rows[-1:],
    'segments':diagnostics['selected_training_segments'],'epochs':epochs,'data':summary_data},ensure_ascii=False))
