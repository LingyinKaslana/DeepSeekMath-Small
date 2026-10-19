"""Read-only CPU reconstruction of the update failing after the last JSONL record."""
import bisect
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

root=Path('.')
run=root/'runs/cpt_10b_20261008'
out=root/'runs/cpt_failure_20261009'
manifest=json.loads((run/'manifest.json').read_text())
data=Path(manifest['arguments']['data'])
with (run/'training.jsonl').open() as f:
    for line in f: last=json.loads(line)
fail_step=last['step']+1
epoch,local=divmod(fail_step-1,manifest['epoch_steps'])
order=np.load(run/'orders'/f'order_epoch_{epoch:05d}.npy',mmap_mode='r')
expected_order=manifest['order_metadata']['files'][epoch]['sha256']
with (run/'orders'/f'order_epoch_{epoch:05d}.npy').open('rb') as f: actual_order=hashlib.file_digest(f,'sha256').hexdigest()
assert actual_order==expected_order
indices=order[local*16:local*16+16].tolist()
tokens=np.load(data/'train_tokens.npy',mmap_mode='r')
tokenizer=AutoTokenizer.from_pretrained(run/'checkpoint_step_00175000',local_files_only=True)
special=set(tokenizer.all_special_ids)

def stats(values):
    vals=np.asarray(values)
    unique,counts=np.unique(vals,return_counts=True)
    p=counts/len(vals)
    adjacent=np.diff(vals.astype(np.int64))==0
    changes=np.flatnonzero(~adjacent)+1
    lengths=np.diff(np.r_[0,changes,len(vals)])
    starts=np.r_[0,changes]
    longest=int(np.argmax(lengths))
    top=np.argsort(counts)[-8:][::-1]
    grams={}
    for n in (4,8):
        counter=collections.Counter(tuple(map(int,vals[i:i+n])) for i in range(len(vals)-n+1))
        gram,count=counter.most_common(1)[0]
        grams[str(n)]={'maximum_occurrences':count,'tokens':list(gram),'decoded':tokenizer.decode(gram)}
    return {'n':len(vals),'minimum_token_id':int(vals.min()),'maximum_token_id':int(vals.max()),
        'unique_tokens':len(unique),'token_entropy_bits':float(-(p*np.log2(p)).sum()),
        'most_common': [{'id':int(unique[i]),'count':int(counts[i]),'fraction':float(counts[i]/len(vals)),
             'decoded':tokenizer.decode([int(unique[i])])} for i in top],
        'longest_same_token_run':int(lengths[longest]),'longest_same_token_run_start':int(starts[longest]),
        'longest_same_token_run_id':int(vals[starts[longest]]),
        'special_counts':{str(int(i)):int(c) for i,c in zip(unique,counts) if int(i) in special},
        'repeated_ngrams':grams}

meta_path=data/'train.jsonl'
meta_size=meta_path.stat().st_size

def documents_overlapping(start,end):
    # Binary search byte offsets, then scan only records adjacent to the target.
    with meta_path.open('rb') as f:
        lo,hi=0,meta_size
        while hi-lo>8192:
            mid=(lo+hi)//2
            f.seek(mid)
            if mid:f.readline()
            pos=f.tell()
            line=f.readline()
            if not line:
                hi=mid
                continue
            rec=json.loads(line)
            if rec['token_offset']<=start:lo=pos
            else:hi=mid
        f.seek(lo)
        selected=[]
        for line in f:
            rec=json.loads(line)
            a=rec['token_offset'];b=a+rec['used_tokens']
            if a>=end:break
            if b>start:selected.append(rec)
        assert selected and selected[0]['token_offset']<=start
        assert selected[-1]['token_offset']+selected[-1]['used_tokens']>=end
        return selected

blocks=[];docs={}
for index in indices:
    start=index*2048;end=min(start+2048,len(tokens))
    values=tokens[start:end]
    records=documents_overlapping(start,end)
    for rec in records:docs[rec['text_sha256']]=rec
    blocks.append({'block_index':index,'start':start,'end':end,'is_tail':len(values)<2048,
        **stats(values),'decoded_prefix':tokenizer.decode(values[:192]),
        'documents':[{'text_sha256':rec['text_sha256'],'url':rec['url'],
            'overlap_tokens':min(end,rec['token_offset']+rec['used_tokens'])-max(start,rec['token_offset'])}
            for rec in records]})

source=json.loads((root/'data/raw/openwebmath_20261008/manifest.json').read_text())
source_dir=Path(source['snapshot'])
grouped=collections.defaultdict(list)
for rec in docs.values():grouped[rec['source_shard'],rec['row_group']].append(rec)
for (shard,group),records in grouped.items():
    table=pq.ParquetFile(source_dir/shard).read_row_group(group,columns=['text','url'])
    for rec in records:
        row=table.slice(rec['row_in_group'],1).to_pylist()[0]
        text=row['text']
        digest=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        ids=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
        actual=np.asarray(ids[:rec['used_tokens']],dtype=np.uint32)
        stored=tokens[rec['token_offset']:rec['token_offset']+rec['used_tokens']]
        rec['source_resolution']={'url_equal':row['url']==rec['url'],
            'normalized_text_sha256_equal':digest==rec['text_sha256'],
            'token_roundtrip_equal':bool(np.array_equal(actual,stored)),
            'original_document_token_count_equal':len(ids)==rec['document_tokens_with_eos'],
            'text_characters':len(text),'text_prefix':text[:400]}

controls=[]
for step in [fail_step-1,fail_step-2,fail_step-10,fail_step-100]:
    e,l=divmod(step-1,manifest['epoch_steps'])
    o=np.load(run/'orders'/f'order_epoch_{e:05d}.npy',mmap_mode='r')
    idx=o[l*16:l*16+16]
    summary=[]
    for i in idx:
        s=stats(tokens[int(i)*2048:min((int(i)+1)*2048,len(tokens))])
        summary.append({k:s[k] for k in ['unique_tokens','token_entropy_bits','longest_same_token_run','special_counts']})
    controls.append({'step':step,'indices':idx.tolist(),'block_statistics':summary})
result={'status':'completed','cpu_only':True,'failure_kind':json.loads((run/'result.json').read_text())['error'],
    'last_successful_record':last,'inferred_failed_update':fail_step,'inference_basis':
      'training.jsonl records every successful update; update raised in clip_grad_norm_ before optimizer and before completed_steps increment',
    'epoch_1_based':epoch+1,'step_in_epoch_1_based':local+1,
    'order_checksum_verified':actual_order,'block_indices':indices,'input_tokens':sum(b['n'] for b in blocks),
    'supervised_tokens':sum(b['n']-1 for b in blocks),'blocks':blocks,'documents':list(docs.values()),
    'comparison_successful_batches':controls,
    'limitations':['No failed-step forward loss or gradient tensors were saved; this CPU report cannot determine the numerical root cause.',
       'Token range/repetition checks cannot prove a mathematically normal model gradient.']}
out.mkdir(parents=True,exist_ok=True)
path=out/'failed_batch_cpu.json'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(path),'failed_update':fail_step,'indices':indices,
 'blocks':len(blocks),'documents':len(docs),'roundtrip_all':all(r['source_resolution']['token_roundtrip_equal'] for r in docs.values()),
 'tail':any(b['is_tail'] for b in blocks),'unique_min':min(b['unique_tokens'] for b in blocks),
 'longest_same_token_run_max':max(b['longest_same_token_run'] for b in blocks),
 'entropy_min':min(b['token_entropy_bits'] for b in blocks),
 'maximum_top_token_fraction':max(b['most_common'][0]['fraction'] for b in blocks)},ensure_ascii=False))
