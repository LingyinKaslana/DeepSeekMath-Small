"""Read-only CPU checks of packing, fixed validation and saved block permutations."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import torch
from transformers import AutoTokenizer

root=Path('.')
data=root/'data/processed/openwebmath_5b_s42'
orig=root/'runs/cpt_10b_20261008'
comparison=root/'runs/cpt_native_compare_20261009'
out=root/'runs/cpt_native_long_20261009'
manifest=json.loads((data/'manifest.json').read_text())
settings=json.loads((data/'settings.json').read_text())
run_manifest=json.loads((orig/'manifest.json').read_text())
plan=json.loads((comparison/'native/plan.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(orig/'checkpoint_step_00175000',local_files_only=True)
eos=tokenizer.eos_token_id
train=np.load(data/'train_tokens.npy',mmap_mode='r')
validation=np.load(data/'validation_tokens.npy',mmap_mode='r')
pilot=np.load(root/'data/processed/openwebmath_10m_s42/validation_tokens.npy',mmap_mode='r')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(values):return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast('B')).hexdigest()

val_docs=[];bad_hash=[];bad_offsets=[];offset=0
with (data/'validation.jsonl').open() as f:
    for line in f:
        row=json.loads(line)
        if row['token_offset']!=offset:bad_offsets.append(row['token_offset'])
        offset+=row['used_tokens']
        if hashlib.sha256(' '.join(row['text'].split()).encode()).hexdigest()!=row['text_sha256']:
            bad_hash.append(row['text_sha256'])
        val_docs.append(row)
assert offset==len(validation)
truncated=[r for r in val_docs if r['prefix_truncated']]
full=[r for r in val_docs if not r['prefix_truncated']]
retokenization=[]
for population in [truncated[:114],truncated[114:],full]:
    for i in sorted(set(np.linspace(0,len(population)-1,6).astype(int).tolist())):
        row=population[i]
        ids=tokenizer.encode(row['text'],add_special_tokens=False)+[eos]
        actual=validation[row['token_offset']:row['token_offset']+row['used_tokens']]
        retokenization.append({'text_sha256':row['text_sha256'],'source_shard':row['source_shard'],
           'prefix_truncated':row['prefix_truncated'],'original_token_count_matches':len(ids)==row['document_tokens_with_eos'],
           'exact_used_token_roundtrip':bool(np.array_equal(np.asarray(ids[:row['used_tokens']],dtype=np.uint32),actual))})

def endpoints(rows,array):
    missing=[r for r in rows if int(array[r['token_offset']+r['used_tokens']-1])!=eos]
    joins=[r for r in missing if r['token_offset']+r['used_tokens']<len(array)]
    in_block=[r for r in joins if (r['token_offset']+r['used_tokens'])%2048!=0]
    affected=sorted({(r['token_offset']+r['used_tokens'])//2048 for r in in_block})
    return {'documents':len(rows),'documents_missing_terminal_eos':len(missing),
       'missing_eos_before_a_following_document':len(joins),'missing_eos_joins_inside_blocks':len(in_block),
       'blocks_containing_missing_eos_joins':len(affected),'affected_block_indices':affected,
       'used_tokens_sum':sum(r['used_tokens'] for r in rows),'used_tokens_lt_64':sum(r['used_tokens']<64 for r in rows)}

meta=data/'train.jsonl';size=meta.stat().st_size
def records_at(start,end):
    with meta.open('rb') as f:
        lo,hi=0,size
        while hi-lo>8192:
            mid=(lo+hi)//2;f.seek(mid)
            if mid:f.readline()
            pos=f.tell();line=f.readline()
            if not line:hi=mid;continue
            row=json.loads(line)
            if row['token_offset']<=start:lo=pos
            else:hi=mid
        f.seek(lo);selected=[]
        for line in f:
            row=json.loads(line);a=row['token_offset'];b=a+row['used_tokens']
            if a>=end:break
            if b>start:selected.append(row)
        assert selected and selected[0]['token_offset']<=start
        return selected

train_tails=[];shard_boundaries=[];offset=0
for shard in manifest['shard_sampling']:
    offset+=shard['tokens']['train'];shard_boundaries.append(offset)
    row=records_at(offset-1,offset)[0]
    assert row['source_shard']==shard['shard'] and row['token_offset']+row['used_tokens']==offset
    train_tails.append(row)
train_endpoints=endpoints(train_tails,train)
train_endpoints['boundary_documents_all_prefix_truncated']=all(r['prefix_truncated'] for r in train_tails)

probe=plan['fixed_train_probe'];probe_indices=np.asarray(probe['indices'],dtype=np.int64)
regenerated=np.sort(np.random.default_rng(probe['seed']).choice(len(train)//2048,size=probe['blocks'],replace=False)).astype(np.int64)
probe_rows=[];probe_hash=hashlib.sha256();probe_documents={};block_entropies=[];probe_eos=0
for index in probe_indices:
    start=int(index)*2048;values=train[start:start+2048]
    probe_hash.update(memoryview(np.ascontiguousarray(values)).cast('B'))
    counts=np.unique(values,return_counts=True)[1];p=counts/len(values)
    block_entropies.append(float(-(p*np.log2(p)).sum()));probe_eos+=int((values==eos).sum())
    recs=records_at(start,start+2048)
    for row in recs:probe_documents[row['text_sha256']]=row
    probe_rows.append({'block_index':int(index),'source_shards':sorted({r['source_shard'] for r in recs}),
                       'document_count_overlapping':len(recs)})
probe_shards={s for row in probe_rows for s in row['source_shards']}
probe_domains=collections.Counter(urlparse(r['url']).netloc for r in probe_documents.values())
vhash={r['text_sha256'] for r in val_docs};vurl={r['url'] for r in val_docs}

orders=[];torch.set_num_threads(2);generator=torch.Generator().manual_seed(42)
blocks=run_manifest['order_metadata']['dataset_blocks']
source_for_block=np.searchsorted(np.asarray(shard_boundaries,dtype=np.int64),np.arange(blocks,dtype=np.int64)*2048,side='right')
overall=np.bincount(source_for_block,minlength=114)
for epoch,item in enumerate(run_manifest['order_metadata']['files']):
    path=orig/'orders'/item['name'];saved=np.load(path,mmap_mode='r')
    exact_permutation=bool(np.array_equal(np.sort(saved),np.arange(blocks,dtype=saved.dtype)))
    rebuilt=torch.randperm(blocks,generator=generator,device='cpu').numpy().astype(saved.dtype)
    coarse=[]
    for start,end in zip(np.linspace(0,blocks,11).astype(int)[:-1],np.linspace(0,blocks,11).astype(int)[1:]):
        counts=np.bincount(source_for_block[saved[start:end]],minlength=114)
        expected=overall*(end-start)/blocks
        nominal_z=(counts-expected)/np.sqrt(expected*(1-overall/blocks)*(1-(end-start)/blocks))
        coarse.append({'order_positions':[int(start),int(end)],'source_shards_present':int((counts>0).sum()),
           'maximum_abs_nominal_shard_count_z':float(np.abs(nominal_z).max())})
    orders.append({'epoch_1_based':epoch+1,'sha256_matches_recorded':sha(path)==item['sha256'],
        'exact_permutation_of_all_blocks':exact_permutation,'exact_seed42_cpu_randperm_reconstruction':bool(np.array_equal(saved,rebuilt)),
        'partial_block_appears_exactly_once':int((saved==blocks-1).sum())==1,
        'position_vs_block_index_pearson':float(np.corrcoef(np.arange(blocks,dtype=np.float64),saved)[0,1]),
        'decile_source_coverage':coarse})
    del rebuilt

comparative_runs={}
for case in ['fast','native']:
    rows=json.loads((comparison/case/'evaluation_history.json').read_text())
    comparative_runs[case]={'validation_initial':rows[0]['validation']['loss'],'validation_final':rows[-1]['validation']['loss'],
       'train_probe_initial':rows[0]['train_probe']['loss'],'train_probe_final':rows[-1]['train_probe']['loss'],
       'extra_tokens':plan['new_input_tokens'],'extra_updates':plan['updates']}
pilot_runs={}
for name in ['cpt_pilot_10m_20261008','cpt_fast_pilot_10m_20261008','cpt_stable_pilot_10m_20261008']:
    r=json.loads((root/'runs'/name/'result.json').read_text())
    pilot_runs[name]={'tokens':r['trained_tokens'],'before':r['validation_before']['loss'],'after':r['validation_after']['loss'],
       'validation_supervised_tokens':r['validation_before']['supervised_tokens']}
initial=json.loads((orig/'validation_before.json').read_text())
base_pilot=pilot_runs['cpt_pilot_10m_20261008']['before']
derived_extension=(initial['loss']*initial['supervised_tokens']-base_pilot*262016)/(initial['supervised_tokens']-262016)
result={'status':'completed','cpu_only':True,'observed_utc':datetime.now(timezone.utc).isoformat(),
    'data_files':{'manifest_sha256':sha(data/'manifest.json'),'validation_tokens_sha256':sha(data/'validation_tokens.npy'),
       'validation_sha_matches_training_and_ab_plans':sha(data/'validation_tokens.npy')==run_manifest['signature']['token_file_sha256']['validation_tokens.npy']==plan['data_artifact_sha256']['validation_tokens.npy'],
       'train_full_20GB_checksum_not_repeated':True,'pilot_validation_prefix_exactly_preserved':bool(np.array_equal(validation[:len(pilot)],pilot))},
    'fixed_split_and_quality':{'train_tokens':len(train),'validation_tokens':len(validation),
       'train_documents':manifest['splits']['train']['documents'],'validation_documents':len(val_docs),
       'text_hash_overlap_recorded':manifest['train_validation_text_hash_overlap'],'url_overlap_recorded':manifest['train_validation_url_overlap'],
       'global_exact_deduplication_recorded':manifest['exact_global_selected_text_and_url_deduplication'],
       'validation_metadata_text_hash_errors':bad_hash,'validation_metadata_offset_errors':bad_offsets,
       'new_validation_records_below_hash_threshold':sum(int(r['text_sha256'][:16],16)<settings['validation_threshold'] for r in val_docs if r['token_offset']>=262144),
       'new_validation_documents':sum(r['token_offset']>=262144 for r in val_docs),
       'pilot_preserved_from_old_threshold_fraction':262144/(10000000+262144),
       'new_threshold_fraction':manifest['validation_fraction'],'retokenization_samples':retokenization,
       'source_roundtrip_basis':'Original preparation fully verified all validation source records/tokens; original manifest sampled 342 train sources; prior failed-batch CPU audit independently verified 23 source documents. This diagnostic does not rescan raw parquet shards.'},
    'packing':{'policy':manifest['packing'],'sequence_length':2048,'eos_token_id':eos,
       'positions':'Reset at every 2048-token block, continuous within block; no document-boundary mask.',
       'targets':'Labels equal token IDs, with causal one-token shift; first token of each block not supervised; padded train tail labels=-100.',
       'validation_full_documents':endpoints(full,validation),'validation_truncated_documents':endpoints(truncated,validation),
       'train_shard_tail_documents':train_endpoints,
       'validation_eos_count':int((validation==eos).sum()),'fixed_train_probe_eos_count':probe_eos,
       'missing_separator_representativeness_note':'Quota-truncated document prefixes lose terminal EOS before the next document. They affect a much larger fraction of validation blocks than train blocks. This is a concrete packing/representativeness issue, not proof of the cause of the loss trend.'},
    'shuffle':{'dataset_same_in_both_epochs':True,'independent_fresh_permutation_each_epoch':True,'audits':orders,
       'interpretation':'Exact regeneration and permutation checks show no wrong seed, duplicated/missing block or file-order curriculum in the saved orders. Nominal decile z-scores are descriptive only.'},
    'fixed_train_probe':{'blocks':len(probe_indices),'input_tokens':probe['input_tokens'],
       'exact_seed_reconstruction':bool(np.array_equal(probe_indices,regenerated)),
       'indices_sha256_matches':array_sha(probe_indices)==probe['indices_sha256'],
       'token_values_sha256_matches':probe_hash.hexdigest()==probe['tokens_sha256'],
       'source_shards_covered':len(probe_shards),'overlapping_documents':len(probe_documents),
       'sample_train_validation_hash_overlap':len(set(probe_documents)&vhash),
       'sample_train_validation_url_overlap':len({r['url'] for r in probe_documents.values()}&vurl),
       'top_domains_by_overlapping_document_count':probe_domains.most_common(12),
       'block_token_entropy_mean':float(np.mean(block_entropies)),
       'representativeness':'Uniform complete-block sample without replacement from the same 5B corpus, excluding the 512-token tail. Good stable corpus probe but only 256 blocks, not all domains and not a held-out set. All blocks were seen during completed epoch one.'},
    'saved_loss_crosschecks':{'base_full_validation_loss':initial['loss'],'pilot_runs':pilot_runs,
       'derived_base_new_validation_extension_loss_approx':derived_extension,
       'derivation_note':'Weighted subtraction using preserved pilot subset and same base-model native evaluations; float32 accumulation/rounding can affect the last digits. Pilot vs full validation averages cannot be compared as if identical datasets.',
       'shared_checkpoint_short_ab':comparative_runs,
       'short_ab_note':'Both arms at 1000 updates improve full-validation and fixed train-probe averages slightly. Their differences are tiny in this short window and do not establish the cause of the prior rise.'},
    'not_established':['No evidence that the fixed validation bytes or block order changed during training.',
       'Validation sampling/prefix truncation can alter representativeness, but fixed bias alone does not establish why loss changes over time.',
       'No benchmark/general-domain evaluation is present. Math-only domain forgetting is possible on other domains, but cannot directly explain worsening on this same-domain held-out set without evidence.',
       'The small uniform train probe is not a complete training-loss census; its behavior alone is not a generalization guarantee.'],
    'minimal_next_steps':['Keep native continuation and existing validation/probe fixed to measure a longer trajectory.',
       'On the same checkpoint, report validation loss separately for preserved pilot blocks, new-extension blocks, and blocks containing missing-EOS quota joins; these are diagnosis slices, not replacements for the existing reference.',
       'If loss degradation concentrates near missing separators, compare an additional properly EOS-terminated, complete-document validation sample from already cached sources. Preserve the current validation reference unchanged.',
       'General-domain forgetting requires a separate fixed code/general-text probe; do not infer it solely from this math-validation curve.']}
out.mkdir(parents=True,exist_ok=True)
(out/'data_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(out/'data_diagnostics.json'),'validation_full_eos':result['packing']['validation_full_documents'],
    'validation_truncated_eos':result['packing']['validation_truncated_documents'],
    'train_tail_eos':train_endpoints,'probe_shards':len(probe_shards),'shuffle':[{k:r[k] for k in ['epoch_1_based','exact_permutation_of_all_blocks','exact_seed42_cpu_randperm_reconstruction','position_vs_block_index_pearson']} for r in orders]},ensure_ascii=False))
