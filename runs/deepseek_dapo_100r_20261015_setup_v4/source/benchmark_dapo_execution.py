"""Isolated full-round microbatch / fixed-prompt rollout throughput comparison."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from dapo_math import load_data,rollout,train_round,write,stamp,sha,cpu_tree,verify_response_log_probs

p=argparse.ArgumentParser();p.add_argument('--setup',type=Path,required=True);a=p.parse_args()
setup=a.setup.resolve();boundary=json.loads((setup/'boundary.json').read_text());checkpoint=Path(boundary['checkpoint'])
run=setup.parent/'deepseek_dapo_100r_20261015';torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
payload=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
config=payload['config'];train,_,_=load_data(Path('data/processed/sft_gsm_math_4096_s42_20261015'))
model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
print(json.dumps({'event':'model_loaded','utc':stamp()}),flush=True)
original=cpu_tree(model.state_dict())
groups=[]
with (run/'rollouts.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        if row['round']==boundary['completed_rounds'] and row['accepted']:groups.append(row['samples'])
assert len(groups)==32
results={'status':'running','checkpoint':str(checkpoint),'resume_round':boundary['completed_rounds'],
         'baseline_config':config,'updates':[],'rollouts':[],'response_alignment':verify_response_log_probs(model,tokenizer,train)}
baseline_samples=None
for micro in (2,8,16,32):
    model.load_state_dict(original);torch.cuda.empty_cache()
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=True,fused=False)
    opt.load_state_dict(payload['optimizer']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    selected={**config,'micro_batch':micro};begin=time.perf_counter()
    try:
        metrics=train_round(model,opt,tokenizer,groups,selected,boundary['completed_rounds'])
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        samples={name:value.detach().flatten()[::max(1,value.numel()//1024)].cpu().clone() for name,value in model.named_parameters()}
        if baseline_samples is None:baseline_samples=samples
        difference=max(float((samples[name]-baseline_samples[name]).abs().max()) for name in samples)
        row={'micro_batch':micro,'seconds':seconds,'response_tokens_per_second':metrics['response_tokens']/seconds,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'updates':metrics['updates'],
             'response_tokens':metrics['response_tokens'],'max_abs_sampled_parameter_diff_vs_micro2':difference,
             'first_minibatch_policy_loss':metrics['metrics'][0]['policy_loss'],'status':'passed'}
        assert difference<1e-5 and metrics['updates']==4
    except torch.OutOfMemoryError:
        row={'micro_batch':micro,'status':'oom'}
    results['updates'].append(row);write(setup/'benchmark.json',results);print(json.dumps({'event':'update_benchmark',**row}),flush=True)
    model.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache()
model.load_state_dict(original)
prompts=train[:32]
for prompt_batch in (16,32):
    torch.manual_seed(4242);torch.cuda.manual_seed_all(4242);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    seconds=0.;tokens=0;correct=0;max_length=0
    try:
        for offset in range(0,len(prompts),prompt_batch):
            values,elapsed=rollout(model,tokenizer,prompts[offset:offset+prompt_batch],config,8)
            seconds+=elapsed
            for group in values:
                tokens+=sum(len(r['tokens']) for r in group);correct+=sum(r['correct'] for r in group)
                max_length=max(max_length,max(len(r['tokens']) for r in group))
        row={'prompt_batch':prompt_batch,'concurrent_responses':prompt_batch*8,'seconds':seconds,
             'generated_tokens':tokens,'tokens_per_second':tokens/seconds,'correct_responses':correct,
             'maximum_response_tokens':max_length,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'status':'passed'}
    except torch.OutOfMemoryError:row={'prompt_batch':prompt_batch,'status':'oom'}
    results['rollouts'].append(row);write(setup/'benchmark.json',results);print(json.dumps({'event':'rollout_benchmark',**row}),flush=True)
    gc.collect();torch.cuda.empty_cache()
successful=[r for r in results['updates'] if r['status']=='passed' and r['peak_allocated_gib']<210]
selected_micro=min(successful,key=lambda r:r['seconds'])['micro_batch']
# A full-cap forward/backward checks memory headroom, not just short-response speed.
model.zero_grad(set_to_none=True)
from dapo_math import log_probs
stress=[{'prompt_ids':train[0]['input_ids'][:train[0]['prompt_length']],
         'tokens':(train[0]['input_ids'][train[0]['prompt_length']:-1]*100)[:2048]} for _ in range(selected_micro)]
assert all(len(row['tokens'])==2048 for row in stress)
torch.cuda.reset_peak_memory_stats()
try:
    current,_=log_probs(model,stress,tokenizer.eos_token_id);(-current.mean()).backward();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/1024**3
    # Includes room for the ~10GiB Adam moments absent in this stress forward.
    results['stress']={'status':'passed','micro_batch':selected_micro,'response_length':2048,'peak_allocated_gib':peak,'adam_headroom_gib':11}
    if peak+11>220:selected_micro=max(2,selected_micro//2)
except torch.OutOfMemoryError:
    results['stress']={'status':'oom','micro_batch':selected_micro};selected_micro=max(2,selected_micro//2)
model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
rollouts=[r for r in results['rollouts'] if r['status']=='passed']
selected_rollout=max(rollouts,key=lambda r:r['tokens_per_second'])['prompt_batch']
results.update(status='complete',finished_utc=stamp(),selected_micro_batch=selected_micro,selected_rollout_prompts=selected_rollout,
               source_checkpoint_unchanged=sha(checkpoint/'checkpoint_manifest.json')==boundary['checkpoint_manifest_sha256'])
write(setup/'benchmark.json',results);print(json.dumps(results),flush=True)
