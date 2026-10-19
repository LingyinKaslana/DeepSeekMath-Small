"""Matched-checkpoint small-concurrency routing/tail benchmarks and update smoke."""
import argparse,json,copy,time,gc,random,math,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from dapo_math import load_data,rollout,train_round,validate,write,stamp
from dapo_tail_sampling import DifficultyRouter
p=argparse.ArgumentParser();p.add_argument('--setup',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--history',type=Path);p.add_argument('--production',action='store_true');p.add_argument('--selected-policy');a=p.parse_args();setup=a.setup.resolve()
checkpoint=(a.checkpoint or Path('runs/deepseek_dapo_10k_20261016/latest_checkpoint')).resolve()
def file_hash(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
source_hashes={f.name:file_hash(f) for f in [checkpoint/'checkpoint_manifest.json',*sorted(checkpoint.glob('*.safetensors'))]}
state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
round_index=state['completed_rounds']
assert round_index>=1900
history=json.loads((a.history or setup/'bench_history/history_audit.json').read_text())['history']
assert max(v['last_round'] for v in history.values())<=round_index
# The paused main process keeps its cache: enforce a smaller isolated budget.
torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.94 if a.production else .29)
model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
train,val,_=load_data(Path('data/processed/sft_gsm_math_4096_s42_20261015'))
config={**state['config'],'rollout_prompts':32 if a.production else 8,'micro_batch':16 if a.production else 8}
policies=tuple(dict.fromkeys(('uniform',a.selected_policy))) if a.production else ('uniform','history_weighted','history_rank','history_tail')
assert None not in policies
result={'status':'running','checkpoint':str(checkpoint),'checkpoint_round':round_index,'concurrent_main_process':'none' if a.production else 'SIGSTOP, state retained, no GPU compute',
        'active_responses':config['rollout_prompts']*8,'production_active_responses':256,'max_new_tokens':2048,
        'limits':['Routing deliberately changes question distribution.','Timing includes candidate generation and selection, excludes model loading and optimizer update.',
                  'Small-concurrency bench is not a production256-concurrency speedup claim.','Tail cancellation deliberately changes rewards of cancelled incomplete responses.'],
        'trials':[]}
write(setup/'benchmark.json',result)
# Warm up native kernels before recording measurements.
torch.cuda.manual_seed_all(4242);rollout(model,tokenizer,train[:config['rollout_prompts']],{**config,'max_new_tokens':64},8)

def collect(policy,trial,target_groups=8):
 cfg={**config,'sampling_policy':policy};router=DifficultyRouter(copy.deepcopy(history),strategy='weighted' if policy=='history_weighted' else 'rank')
 epoch,local=divmod(state['training_order_cursor'],len(train));order=list(range(len(train)));random.Random(42+epoch).shuffle(order)
 cursor=local;accepted=[];candidates=0;tokens=0;found=0;cancelled=0;raw_mixed=0;selected_lengths=[];counts={'all_wrong':0,'all_correct':0,'mixed':0}
 def next_pool(number):
  nonlocal cursor,epoch,order
  rows=[]
  for _ in range(number):
   if cursor>=len(train):
    epoch+=1;order=list(range(len(train)));random.Random(42+epoch).shuffle(order);cursor=0
   rows.append(train[order[cursor]]);cursor+=1
  return rows
 torch.cuda.manual_seed_all(42+trial);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 while len(accepted)<target_groups:
  assert candidates<1024,'Benchmark mixed groups exhausted'
  number=config['rollout_prompts']
  pool=next_pool(number if policy=='uniform' else number*8)
  rows=pool if policy=='uniform' else router.choose(pool,number,seed=42000+trial*10000+candidates)
  groups,elapsed=rollout(model,tokenizer,rows,cfg,8)
  for group in groups:
   candidates+=1;n=sum(r['correct'] for r in group);raw=sum(r.get('raw_correct',r['correct']) for r in group)
   mixed=0<n<8;found+=mixed;raw_mixed+=0<raw<8
   counts['mixed' if mixed else 'all_correct' if n==8 else 'all_wrong']+=1
   tokens+=sum(len(r['tokens']) for r in group);cancelled+=sum(r.get('forced_incorrect',False) for r in group)
   router.observe(group,round_index+1)
   if mixed and len(accepted)<target_groups:accepted.append(group);selected_lengths.extend(len(r['tokens']) for r in group)
  progress={'event':'benchmark_batch','utc':stamp(),'policy':policy,'trial':trial,'candidates':candidates,'used_groups':len(accepted),'seconds':time.perf_counter()-started}
  print(json.dumps(progress),flush=True);write(setup/'benchmark_progress.json',progress)
 seconds=time.perf_counter()-started
 summary={'policy':policy,'trial':trial,'seconds':seconds,'used_groups':len(accepted),'found_groups':found,'candidates':candidates,
          'effective_groups_per_second':len(accepted)/seconds,'generated_tokens':tokens,'mixed_fraction':found/candidates,
          'raw_mixed_groups':raw_mixed,'forced_incorrect_responses':cancelled,'forced_fraction':cancelled/(candidates*8),
          'mean_used_response_tokens':sum(selected_lengths)/len(selected_lengths),'counts':counts,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
 return summary,accepted

for trial in range(1 if a.production else 2):
 for policy in policies if trial==0 else tuple(reversed(policies)):
  summary,_=collect(policy,trial,target_groups=32 if a.production else 8);result['trials'].append(summary);write(setup/'benchmark.json',result)
  print(json.dumps({'event':'benchmark_trial',**summary}),flush=True)
summary={}
for policy in policies:
 trials=[r for r in result['trials'] if r['policy']==policy]
 summary[policy]={'effective_groups_per_second':sum(r['used_groups'] for r in trials)/sum(r['seconds'] for r in trials),
                  'mean_mixed_fraction':sum(r['found_groups'] for r in trials)/sum(r['candidates'] for r in trials),
                  'forced_fraction':sum(r['forced_incorrect_responses'] for r in trials)/(sum(r['candidates'] for r in trials)*8)}
baseline=summary['uniform']['effective_groups_per_second']
for value in summary.values():value['speedup_vs_uniform']=value['effective_groups_per_second']/baseline
eligible=[p for p in policies if summary[p]['forced_fraction']<=.05]
winner=max(eligible,key=lambda p:summary[p]['effective_groups_per_second'])
if summary[winner]['speedup_vs_uniform']<1.15:winner='uniform'
# Keep tail guard only if it materially beats routing alone.
if winner=='history_tail' and 'history_rank' in summary and summary[winner]['effective_groups_per_second']<1.1*summary['history_rank']['effective_groups_per_second']:winner='history_rank'
result.update(summary=summary,selected_policy=winner);write(setup/'benchmark.json',result)
_,accepted=collect(winner,3,target_groups=32)
# Verify real 4-update/minibatch64 training, restoring original full Adam state.
optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=True,fused=False)
optimizer.load_state_dict(state['optimizer']);del state;gc.collect();torch.cuda.empty_cache()
probe=random.Random(2042).sample(val,256)[:64]
before=validate(model,tokenizer,probe,config,setup/'smoke_validation_before.jsonl')
metrics=train_round(model,optimizer,tokenizer,accepted,config,round_index)
assert metrics['updates']==4 and all(math.isfinite(r['gradient_norm']) and r['gradient_norm']>0 for r in metrics['metrics'])
after=validate(model,tokenizer,probe,config,setup/'smoke_validation_after.jsonl')
# Small probe is a guard against catastrophic execution bugs, not proof of quality preservation.
assert after['correct']>=before['correct']-8,'Probe regressed sharply; do not deploy automatically'
for name,digest in source_hashes.items():assert file_hash(checkpoint/name)==digest,name
result.update(source_checkpoint_hashes=source_hashes)
result.update(status='complete',finished_utc=stamp(),update_smoke=metrics,probe_before=before,probe_after=after,
              smoke_updates_discarded=True,source_checkpoint_unchanged=True)
write(setup/'benchmark.json',result);print(json.dumps(result),flush=True)
