from pathlib import Path
import hashlib,json,time
from datetime import datetime,timezone
import torch
from safetensors import safe_open
from transformers import AutoConfig,AutoModelForCausalLM
root=Path.cwd();run=root/'runs/deepseek_dapo_100r_20261015';directory=root/'runs/deepseek_dapo_recovery_20261015'
p=run/'latest_checkpoint.incomplete';torch.set_num_threads(8)
m=json.loads((p/'checkpoint_manifest.json').read_text())
s=torch.load(p/'training_state.pt',map_location='cpu',weights_only=True)
assert {k:v for k,v in s.items() if k not in ('optimizer','cpu_rng','cuda_rng')}==m
assert m['completed_rounds']==60 and m['gradient_updates']==240 and m['accepted_responses']==15360
assert m['validation_history'][-1]['round']==60
with (run/'training.jsonl').open() as f:rows=[json.loads(l) for l in f]
assert len(rows)==60 and [r['round'] for r in rows]==list(range(1,61))
assert m['trained_response_tokens']==sum(r['response_tokens'] for r in rows)
assert m['sampled_response_tokens']==sum(r['sampled_response_tokens'] for r in rows)
with torch.device('meta'):model=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(p,local_files_only=True))
expected=model.state_dict();seen=set();weights_sha={}
for shard in sorted(p.glob('*.safetensors')):
 with safe_open(shard,framework='pt',device='cpu') as reader:
  for key in reader.keys():
   value=reader.get_tensor(key)
   assert key not in seen and tuple(value.shape)==tuple(expected[key].shape)
   assert value.dtype==torch.float32 and torch.isfinite(value).all(),key
   seen.add(key)
 with shard.open('rb') as f:weights_sha[shard.name]=hashlib.file_digest(f,'sha256').hexdigest()
assert seen==set(expected) and len(seen)==219
parameters=list(model.named_parameters());opt=s['optimizer'];assert len(opt['state'])==len(parameters)==219
for index,(name,value) in enumerate(parameters):
 state=opt['state'][index]
 assert float(state['step'])==240
 for key in ('exp_avg','exp_avg_sq'):
  tensor=state[key];assert tensor.dtype==torch.float32 and tuple(tensor.shape)==tuple(value.shape) and torch.isfinite(tensor).all(),name
for name in ('cpu_rng','cuda_rng'):assert s[name].dtype==torch.uint8 and s[name].numel()>0
assert opt['param_groups'][0]['lr']==1e-6
record={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'recovered_round':60,'updates':240,'accepted_responses':15360,'weight_tensors':219,'optimizer_states':219,'weights_sha256':weights_sha,'rng_present':True,'training_counters_match_all60_logs':True,'validation_correct':81,'validation_total':256,'reason':'All checkpoint files were completely serialized before stale-backup assertion, prior to atomic directory commit'}
(directory/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'latest_checkpoint').rename(directory/'checkpoint55_retained')
(run/'latest_checkpoint.previous').rename(directory/'checkpoint50_stale_backup_retained')
p.rename(run/'latest_checkpoint')
record['promoted_checkpoint']=str(run/'latest_checkpoint')
(directory/'recovery.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
