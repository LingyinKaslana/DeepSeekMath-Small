"""GPU equivalence, controlled long-tail throughput, and policy-update smoke."""
import argparse,json,gc,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from compact_decode import compact_generate
from dapo_math import load_data,rollout,train_round,write,stamp

p=argparse.ArgumentParser();p.add_argument('--setup',type=Path,required=True);a=p.parse_args()
setup=a.setup.resolve();boundary=json.loads((setup/'boundary.json').read_text());checkpoint=Path(boundary['checkpoint'])
torch.set_num_threads(8);tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
config=state['config'];train,_,_=load_data(Path('data/processed/sft_gsm_math_4096_s42_20261015'))
model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
print(json.dumps({'event':'model_loaded','utc':stamp()}),flush=True)
def trim(values):
    rows=[]
    for row in values.cpu().tolist():rows.append(row[:row.index(tokenizer.eos_token_id)+1] if tokenizer.eos_token_id in row else row)
    return rows
inputs=tokenizer([r['prompt'] for r in train[:8]],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
generation=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,num_beams=1,max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    torch.cuda.manual_seed_all(123)
    reference=model.generate(**inputs,generation_config=generation,logits_to_keep=1)[:,inputs['input_ids'].shape[1]:]
    torch.cuda.manual_seed_all(123)
    manual=compact_generate(model,**inputs,eos_token_id=tokenizer.eos_token_id,max_new_tokens=128,compact=False)
assert trim(reference)==trim(manual),'Uncompacted sampler differs from native HF'
generation.do_sample=False;generation.temperature=None;generation.top_p=None;generation.top_k=None
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    reference=model.generate(**inputs,generation_config=generation,logits_to_keep=1)[:,inputs['input_ids'].shape[1]:]
    compacted=compact_generate(model,**inputs,eos_token_id=tokenizer.eos_token_id,max_new_tokens=128,do_sample=False)
old,new=trim(reference),trim(compacted)
matches=sum(x==y for x,y in zip(old,new));first_matches=sum(x[0]==y[0] for x,y in zip(old,new))
assert first_matches==8 and matches>=7, 'Inspect compacted greedy discrepancy before deployment'
result={'status':'running','checkpoint_round':boundary['completed_rounds'],'uncompacted_sampling_exact_hf':True,
    'greedy_exact_sequences':matches,'greedy_sequences':8,'greedy_first_tokens_exact':first_matches,'cases':[]}
write(setup/'benchmark.json',result)
# Force only row0 to continue to the cap; all other rows can terminate normally.
# Row0 stays row0 after compaction because it never ends before the cap.
def force_long_row(module,args,output):
    output[0,-1,tokenizer.eos_token_id]=-torch.inf
    return output
handle=model.lm_head.register_forward_hook(force_long_row)
bench_config={**config,'max_new_tokens':512,'soft_start':384}
for compact in (False,True):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.manual_seed_all(4242)
    values,seconds=rollout(model,tokenizer,train[:32],{**bench_config,'compact_decode':compact},8)
    lengths=[len(row['tokens']) for group in values for row in group]
    row={'compact':compact,'case':'One of256 responses forced to512 tokens; others stop on EOS',
        'seconds':seconds,'responses':len(lengths),'generated_tokens':sum(lengths),
        'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    assert max(lengths)==512
    result['cases'].append(row);write(setup/'benchmark.json',result);print(json.dumps({'event':'long_tail_benchmark',**row}),flush=True)
    del values;gc.collect()
handle.remove()
torch.cuda.manual_seed_all(42)
values,seconds=rollout(model,tokenizer,train[:32],{**config,'compact_decode':True},8)
mixed=[group for group in values if 0<sum(row['correct'] for row in group)<8]
assert len(mixed)>=4
optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=True,fused=False)
optimizer.load_state_dict(state['optimizer'])
metrics=train_round(model,optimizer,tokenizer,mixed[:4],{**config,'micro_batch':8,'update_batch':8},boundary['completed_rounds'])
assert metrics['updates']==4 and all(row['gradient_norm']>0 for row in metrics['metrics'])
result['update_smoke']=metrics
speedup=result['cases'][0]['seconds']/result['cases'][1]['seconds']
result.update(status='complete',finished_utc=stamp(),long_tail_speedup=speedup,selected_compact_decode=speedup>=1.1,
    diagnostic_updates_discarded=True,formal_resume_from_original_checkpoint=True,
    precision_unchanged='FP32 parameters, BF16 autocast, original mixed-dtype dynamic KV cache',
    limits=['Compaction changes stochastic RNG consumption and BF16 batch-shape rounding.',
            'Controlled long-tail case is a performance test, not an end-to-end training speedup claim.'])
write(setup/'benchmark.json',result);print(json.dumps(result),flush=True)
