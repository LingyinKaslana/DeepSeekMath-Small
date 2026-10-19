"""Small, single-device DAPO: native HF rollouts, full-parameter PyTorch updates.

No KL loss, critic, reward model, LoRA, replay across rollout rounds or top-p
truncation. Old log probabilities are recomputed before *any* round update.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def append(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def numeric_answer(text):
    # First explicit answer marker wins: later guesses cannot override it.
    matches=list(re.finditer(r'(?:^|\n)\s*(?:the[ \t]+answer[ \t]+is[ \t]*|####[ \t]*)([^\n]+)',text,re.I))
    if not matches:return None
    value=matches[0].group(1).strip().replace(',','')
    match=re.fullmatch(r'\$?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\$?\s*\.?',value)
    if not match:return None
    try:return str(Decimal(match.group(1)).normalize())
    except InvalidOperation:return None


def grade(text, expected):
    value=numeric_answer(text)
    return value is not None and Decimal(value)==Decimal(expected)


def shaped_reward(correct, length, soft_start=1536, cap=2048):
    return (1. if correct else -1.)-max(0.,min(1.,(length-soft_start)/(cap-soft_start)))


def advantages(rewards):
    values=torch.tensor(rewards,dtype=torch.float32)
    std=values.std(correction=0)
    return (values-values.mean())/(std+1e-6)


def surrogate(new, old, advantage, low=.2, high=.28):
    delta=new-old;ratio=delta.exp()
    clipped=ratio.clamp(1-low,1+high)
    return -torch.minimum(ratio*advantage,clipped*advantage),ratio


def read_rows(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream]


def load_data(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['status']=='ready'
    for name,info in manifest['artifacts'].items():assert sha(directory/name)==info['sha256']
    train=[r for r in read_rows(directory/'train.jsonl') if r['dataset']=='gsm8k']
    validation=[r for r in read_rows(directory/'validation.jsonl') if r['dataset']=='gsm8k']
    assert len(train)==7217 and len(validation)==256
    assert not {r['question_sha256'] for r in train}&{r['question_sha256'] for r in validation}
    for row in train+validation:
        # The preparer appends this final line from GSM8K's #### gold field.
        # Earlier occurrences in the worked solution are not the gold label.
        answer=numeric_answer('The answer is '+row['completion'].rsplit('The answer is ',1)[1])
        assert answer is not None
        row['expected']=answer
    return train,validation,manifest


@torch.inference_mode()
def rollout(model, tokenizer, rows, config, samples=8):
    model.eval();tokenizer.padding_side='left'
    inputs=tokenizer([row['prompt'] for row in rows],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
    generation=GenerationConfig(do_sample=samples>1,temperature=1. if samples>1 else None,
        top_p=1. if samples>1 else None,top_k=0 if samples>1 else None,
        num_return_sequences=samples,num_beams=1,max_new_tokens=config['max_new_tokens'],
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,use_cache=True)
    begin=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if config.get('compact_decode',False):
            from compact_decode import compact_generate
            generated=compact_generate(model,inputs['input_ids'].repeat_interleave(samples,dim=0),
                inputs['attention_mask'].repeat_interleave(samples,dim=0),tokenizer.eos_token_id,
                config['max_new_tokens'],do_sample=samples>1)
        else:
            generated=model.generate(**inputs,generation_config=generation,logits_to_keep=1)
            generated=generated[:,inputs['input_ids'].shape[1]:]
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    generated=generated.cpu().tolist()
    groups=[]
    for index,row in enumerate(rows):
        group=[]
        for tokens in generated[index*samples:(index+1)*samples]:
            ended=tokenizer.eos_token_id in tokens
            if ended:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
            text=tokenizer.decode(tokens,skip_special_tokens=True)
            correct=grade(text,row['expected'])
            group.append({'source_id':row['source_id'],'question_sha256':row['question_sha256'],
                'prompt_ids':row['input_ids'][:row['prompt_length']], 'tokens':tokens,
                'completion':text,'expected':row['expected'],'extracted':numeric_answer(text),
                'correct':bool(correct),'finish_reason':'eos' if ended else 'length',
                'reward':shaped_reward(correct,len(tokens),config['soft_start'],config['max_new_tokens'])})
        groups.append(group)
    return groups,seconds


def log_probs(model, examples, eos):
    """Return response-only causal log-probs/entropy; right-padding never enters loss."""
    lengths=[len(r['prompt_ids'])+len(r['tokens']) for r in examples]
    width=math.ceil(max(lengths)/8)*8
    ids=torch.full((len(examples),width),eos,dtype=torch.long,device='cuda')
    mask=torch.zeros_like(ids)
    for i,r in enumerate(examples):
        values=r['prompt_ids']+r['tokens']
        ids[i,:len(values)]=torch.tensor(values,device='cuda');mask[i,:len(values)]=1
    with torch.autocast('cuda',dtype=torch.bfloat16):
        # Select response positions *before* the vocabulary projection.
        # LlamaModel is the native HF backbone; lm_head and CE remain native.
        hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        selected=torch.cat([hidden[i,len(r['prompt_ids'])-1:lengths[i]-1] for i,r in enumerate(examples)])
        logits=model.lm_head(selected).float()
    targets=torch.cat([torch.tensor(r['tokens'],device='cuda') for r in examples])
    logp=-F.cross_entropy(logits,targets,reduction='none')
    with torch.no_grad():
        logz=torch.logsumexp(logits,dim=-1)
        entropy=logz-(logits.softmax(-1)*logits).sum(-1)
    return logp,entropy


def train_round(model,optimizer,tokenizer,groups,config,round_index):
    examples=[]
    for group in groups:
        assert 0<sum(r['correct'] for r in group)<len(group)
        adv=advantages([r['reward'] for r in group]).tolist()
        for row,value in zip(group,adv):row['advantage']=value;examples.append(row)
    model.eval()  # No stochastic dropout between behavior/current log probabilities.
    ordered=sorted(examples,key=lambda r:len(r['prompt_ids'])+len(r['tokens']))
    with torch.inference_mode():
        for offset in range(0,len(ordered),config['micro_batch']):
            batch=ordered[offset:offset+config['micro_batch']]
            old,_=log_probs(model,batch,tokenizer.eos_token_id)
            cursor=0
            for row in batch:
                count=len(row['tokens']);row['old_logp']=old[cursor:cursor+count].cpu().clone();cursor+=count
    random.Random(config['seed']+round_index).shuffle(examples)
    stats=[]
    for offset in range(0,len(examples),config['update_batch']):
        minibatch=sorted(examples[offset:offset+config['update_batch']],key=lambda r:len(r['prompt_ids'])+len(r['tokens']))
        denominator=sum(len(row['tokens']) for row in minibatch)
        optimizer.zero_grad(set_to_none=True)
        sums=torch.zeros(5,device='cuda',dtype=torch.float64)
        for start in range(0,len(minibatch),config['micro_batch']):
            batch=minibatch[start:start+config['micro_batch']]
            current,entropy=log_probs(model,batch,tokenizer.eos_token_id)
            old=torch.cat([r['old_logp'] for r in batch]).to('cuda')
            adv=torch.cat([torch.full((len(r['tokens']),),r['advantage'],device='cuda') for r in batch])
            losses,ratio=surrogate(current,old,adv,config['clip_low'],config['clip_high'])
            loss=losses.sum()/denominator
            assert torch.isfinite(loss),'Nonfinite policy loss'
            loss.backward()
            with torch.no_grad():
                delta=current-old
                # k3 estimator, diagnostic only: no KL regularization.
                k3=(-delta).exp()-1+delta
                sums+=torch.stack([losses.sum(),entropy.sum(),k3.sum(),
                    ((ratio<1-config['clip_low'])|(ratio>1+config['clip_high'])).sum(),ratio.sum()]).double()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        values=(sums/denominator).cpu().tolist()
        stats.append(dict(zip(('policy_loss','entropy','old_policy_kl_estimate','clip_fraction','mean_ratio'),values),
                          gradient_norm=float(norm),response_tokens=denominator))
    optimizer.zero_grad(set_to_none=True)
    return {'updates':len(stats),'response_tokens':sum(len(r['tokens']) for r in examples),
            'input_tokens':sum(len(r['tokens'])+len(r['prompt_ids']) for r in examples),
            'metrics':stats}


@torch.inference_mode()
def verify_response_log_probs(model,tokenizer,rows):
    """Compare selected-position log probs with HF's independent masked CE path."""
    row=min(rows,key=lambda r:len(r['input_ids']))
    example={'prompt_ids':row['input_ids'][:row['prompt_length']],
             'tokens':row['input_ids'][row['prompt_length']:]}
    selected,_=log_probs(model,[example],tokenizer.eos_token_id)
    ids=torch.tensor([row['input_ids']],device='cuda')
    labels=ids.clone();labels[:,:row['prompt_length']]=-100
    with torch.autocast('cuda',dtype=torch.bfloat16):
        reference=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
    difference=abs(float(-selected.mean()-reference))
    assert difference<.03, f'Response alignment/CE mismatch: {difference}'
    return {'status':'passed','response_tokens':len(example['tokens']),
            'selected_response_nll':float(-selected.mean()),'native_hf_answer_only_ce':float(reference),
            'absolute_difference':difference,'tolerance':.03}


@torch.inference_mode()
def validate(model,tokenizer,rows,config,path):
    # Preserve training RNG: greedy validation must not perturb rollout randomness.
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    records=[];seconds=0.
    for offset in range(0,len(rows),32):
        groups,elapsed=rollout(model,tokenizer,rows[offset:offset+32],config,samples=1)
        records.extend(group[0] for group in groups);seconds+=elapsed
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
    for row in records:append(path,row)
    return {'examples':len(records),'correct':sum(r['correct'] for r in records),
            'accuracy':sum(r['correct'] for r in records)/len(records),
            'truncated':sum(r['finish_reason']=='length' for r in records),'seconds':seconds,
            'protocol':'GSM8K held-out 0-shot SFT prompt; greedy; EOS only; strict numeric marker grader'}


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path,model,tokenizer,optimizer,state):
    tmp=path.with_name(path.name+'.incomplete');assert not tmp.exists();tmp.mkdir()
    model.save_pretrained(tmp,safe_serialization=True,max_shard_size='4GB');tokenizer.save_pretrained(tmp)
    torch.save({**state,'optimizer':cpu_tree(optimizer.state_dict()),'cpu_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state()},tmp/'training_state.pt')
    write(tmp/'checkpoint_manifest.json',state)
    # Unique backup names tolerate interruption after a prior atomic commit.
    # Only remove this invocation's backup; retained recovery artifacts stay intact.
    backup=path.with_name(path.name+f'.previous.{time.time_ns()}');assert not backup.exists()
    if path.exists():path.rename(backup)
    tmp.rename(path)
    if backup.exists():shutil.rmtree(backup)
    gc.collect()


def checkpoint_check(path,model,optimizer):
    from safetensors import safe_open
    count=0
    current=model.state_dict()
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as reader:
            for key in reader.keys():
                value=reader.get_tensor(key)
                assert torch.isfinite(value).all() and torch.equal(value,current[key].detach().cpu()),key
                count+=1
    assert count==len(current)
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
    expected=optimizer.state_dict();loaded=state['optimizer'];assert loaded['param_groups']==expected['param_groups']
    for index,item in expected['state'].items():
        for key,value in item.items():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all() and torch.equal(value.cpu(),loaded['state'][index][key])
    return {'status':'passed','weight_tensors':count,'optimizer_states':len(loaded['state']),
            'completed_rounds':state['completed_rounds'],'gradient_updates':state['gradient_updates']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--groups',type=int,default=32);p.add_argument('--samples',type=int,default=8)
    p.add_argument('--rollout-prompts',type=int,default=16);p.add_argument('--micro-batch',type=int,default=2)
    p.add_argument('--update-batch',type=int,default=64);p.add_argument('--validation-size',type=int,default=256)
    p.add_argument('--diagnostic-prompts',type=int,default=256);p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--compact-decode',action='store_true')
    p.add_argument('--allow-execution-change',action='store_true',
                   help='Explicitly permit only micro/rollout batch and source version changes on resume')
    a=p.parse_args();assert a.rounds>0 and a.groups>0 and a.samples>1
    a.output.mkdir(parents=True,exist_ok=a.resume is not None)
    torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    train,validation,data_manifest=load_data(a.data)
    parent=json.loads((a.checkpoint/'checkpoint_manifest.json').read_text())
    assert parent['stage']=='sft' and parent['completed_steps']==904 and not parent['diagnostic_smoke']
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    config={'algorithm':'DAPO','rounds':a.rounds,'groups':a.groups,'samples':a.samples,
        'rollout_prompts':a.rollout_prompts,'micro_batch':a.micro_batch,'update_batch':a.update_batch,
        'compact_decode':a.compact_decode,
        'compact_decode_sha256':sha(Path(__file__).with_name('compact_decode.py')) if a.compact_decode else None,
        'max_new_tokens':2048,'soft_start':1536,'clip_low':.2,'clip_high':.28,'lr':1e-6,
        'warmup_rounds':min(10,a.rounds),'seed':42,'max_candidate_groups_per_round':a.groups*16,
        'validation_size':a.validation_size,'validation_every':10,'save_every':5,
        'precision':'FP32 parameters/gradients/Adam; BF16 autocast; native SDPA',
        'kl_coefficient':0.,'temperature':1.,'top_p':1.,'top_k':0,'epochs_per_rollout':1,
        'dynamic_sampling':'Keep groups with both correct and incorrect responses; resample within bounded attempts',
        'reward':'Strict first explicit numeric answer marker: correct +1, incorrect -1; soft length penalty 1536..2048',
        'prompt':'Existing zero-shot Q/A SFT prompt; no few-shot or chat template',
        'source_checkpoint':str(a.checkpoint.resolve()),'source_checkpoint_manifest_sha256':sha(a.checkpoint/'checkpoint_manifest.json'),
        'source_weights_sha256':{f.name:sha(f) for f in sorted(a.checkpoint.glob('*.safetensors'))},
        'data_manifest_sha256':sha(a.data/'manifest.json'),'script_sha256':sha(Path(__file__))}
    state={'stage':'rl','algorithm':'DAPO','diagnostic_smoke':a.smoke,'config':config,
        'completed_rounds':0,'gradient_updates':0,'candidate_groups':0,'sampled_responses':0,
        'sampled_response_tokens':0,'accepted_responses':0,'trained_response_tokens':0,'trained_input_tokens':0,
        'training_order_cursor':0,'best_validation_accuracy':-1.,'best_round':None,'validation_history':[],
        'source_cpt_input_tokens':10_000_000_000,'source_sft_input_tokens':parent['sft_input_tokens']}
    invocation_manifest=a.output/'manifest.json' if not a.resume else a.output/f'resume_manifest_{time.time_ns()}.json'
    write(invocation_manifest,{'started_utc':stamp(),'config':config,'training_examples':len(train),
        'validation_examples':min(a.validation_size,len(validation)),'diagnostic_smoke':a.smoke,
        'optimizer_initialization':('Restore checkpoint AdamW moments and RNG' if a.resume else 'Fresh AdamW, no SFT moments'),
        'resume_checkpoint':str(a.resume.resolve()) if a.resume else None,
        'data_sources':data_manifest['source_files'] if 'source_files' in data_manifest else [],
        'paper_differences':['1.3B SFT initialization; GSM8K only; 8 responses/group; 32 effective groups/round',
            '2048 generation cap/512 soft buffer; native single-device HF; FP32 master parameters',
            'Warmup 10 rounds; 64-response optimizer minibatches; +/-1 numeric correctness reward',
            'GSM8K was already used in SFT; not unseen-task generalization; CPT decontamination incomplete']} )
    model=AutoModelForCausalLM.from_pretrained(a.resume or a.checkpoint,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    assert model.config.attention_dropout==0.
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.999),eps=1e-8,
                                weight_decay=0.,foreach=True,fused=False)
    if a.resume:
        restored=torch.load(a.resume/'training_state.pt',map_location='cpu',weights_only=True)
        assert restored['diagnostic_smoke']==a.smoke
        if a.allow_execution_change:
            allowed={'micro_batch','rollout_prompts','script_sha256','compact_decode','compact_decode_sha256'}
            assert {k:v for k,v in restored['config'].items() if k not in allowed}=={k:v for k,v in config.items() if k not in allowed}
            restored.setdefault('execution_changes',[]).append({'utc':stamp(),'after_round':restored['completed_rounds'],
                'old_config':restored['config'],'new_config':config,'policy_hyperparameters_unchanged':True})
            restored['config']=config
        else:assert restored['config']==config
        optimizer.load_state_dict(restored.pop('optimizer'));torch.set_rng_state(restored.pop('cpu_rng'))
        torch.cuda.set_rng_state(restored.pop('cuda_rng'));state=restored
    print(json.dumps({'event':'model_loaded','utc':stamp(),'parameters':sum(p.numel() for p in model.parameters()),'config':config}),flush=True)
    if a.smoke:
        alignment=verify_response_log_probs(model,tokenizer,train)
        write(a.output/'response_alignment_verification.json',alignment)
        print(json.dumps({'event':'response_alignment_verified',**alignment}),flush=True)
    if a.diagnostic_prompts and not a.resume:
        selected=random.Random(1042).sample(train,a.diagnostic_prompts);counts=Counter();seconds=0.;tokens=0
        for offset in range(0,len(selected),a.rollout_prompts):
            groups,elapsed=rollout(model,tokenizer,selected[offset:offset+a.rollout_prompts],config,a.samples)
            seconds+=elapsed
            for group in groups:
                n=sum(r['correct'] for r in group);counts['groups']+=1
                counts['all_wrong' if n==0 else 'all_correct' if n==a.samples else 'mixed']+=1
                counts['correct_responses']+=n;counts['truncated']+=sum(r['finish_reason']=='length' for r in group)
                tokens+=sum(len(r['tokens']) for r in group)
                append(a.output/'diagnostic_samples.jsonl',group)
            result={'status':'running','groups':dict(counts),'responses':counts['groups']*a.samples,
                    'generated_tokens':tokens,'generation_seconds':seconds,'utc':stamp()}
            write(a.output/'diagnostic.json',result);print(json.dumps({'event':'diagnostic_progress',**result}),flush=True)
        result.update(status='complete',effective_group_fraction=counts['mixed']/counts['groups'])
        write(a.output/'diagnostic.json',result)
        assert counts['mixed']>0,'No mixed-reward groups; cannot start DAPO'
    if a.diagnostic_only:
        write(a.output/'result.json',{'status':'diagnostic_complete','training_updates':0});return
    validation=random.Random(2042).sample(validation,min(a.validation_size,len(validation)))
    if not a.resume:
        before=validate(model,tokenizer,validation,config,a.output/'validation_round000.jsonl')
        state['validation_history'].append({'round':0,**before});state['best_validation_accuracy']=before['accuracy']
        state['best_round']=0;write(a.output/'validation_before.json',before)
        print(json.dumps({'event':'validation_before',**before}),flush=True)
        save_checkpoint(a.output/'best_checkpoint',model,tokenizer,optimizer,state)
        save_checkpoint(a.output/'latest_checkpoint',model,tokenizer,optimizer,state)
    def next_rows(number):
        rows=[]
        for _ in range(number):
            epoch,local=divmod(state['training_order_cursor'],len(train))
            if epoch not in orders:
                indices=list(range(len(train)));random.Random(42+epoch).shuffle(indices);orders[epoch]=indices
            rows.append(train[orders[epoch][local]]);state['training_order_cursor']+=1
        return rows
    orders={};start=time.perf_counter()
    while state['completed_rounds']<a.rounds:
        round_index=state['completed_rounds'];accepted=[];candidates=0;local_tokens=0;rollout_seconds=0.;counts=Counter()
        while len(accepted)<a.groups:
            budget=config['max_candidate_groups_per_round']-candidates
            assert budget>0,'Dynamic sampling exhausted bounded candidate budget; checkpoint retained'
            write(a.output/'progress.json',{'status':'sampling','utc':stamp(),'completed_rounds':round_index,
                'round':round_index+1,'target_rounds':a.rounds,'effective_groups':len(accepted),
                'target_groups':a.groups,'candidate_groups':candidates,'batch_in_flight':True,
                'allocated_gib':torch.cuda.memory_allocated()/1024**3,
                'reserved_gib':torch.cuda.memory_reserved()/1024**3})
            groups,elapsed=rollout(model,tokenizer,next_rows(min(a.rollout_prompts,budget)),config,a.samples)
            rollout_seconds+=elapsed
            for group in groups:
                candidates+=1;correct=sum(r['correct'] for r in group)
                category='all_wrong' if correct==0 else 'all_correct' if correct==a.samples else 'mixed'
                keep=category=='mixed' and len(accepted)<a.groups
                counts[category]+=1;counts['correct_responses']+=correct
                counts['truncated_responses']+=sum(r['finish_reason']=='length' for r in group)
                local_tokens+=sum(len(r['tokens']) for r in group)
                append(a.output/'rollouts.jsonl',{'round':round_index+1,'accepted':keep,'samples':group})
                if keep:accepted.append(group)
            progress={'status':'sampling','utc':stamp(),'completed_rounds':round_index,'target_rounds':a.rounds,
                'round':round_index+1,'effective_groups':len(accepted),'target_groups':a.groups,
                'candidate_groups':candidates,'counts':dict(counts),'rollout_seconds':rollout_seconds,
                'generation_tokens_per_second':local_tokens/max(rollout_seconds,1e-9),
                'allocated_gib':torch.cuda.memory_allocated()/1024**3,
                'reserved_gib':torch.cuda.memory_reserved()/1024**3}
            write(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        rate=config['lr']*min(1.,(round_index+1)/config['warmup_rounds'])
        for param_group in optimizer.param_groups:param_group['lr']=rate
        write(a.output/'progress.json',{**progress,'status':'updating'})
        update_start=time.perf_counter();metrics=train_round(model,optimizer,tokenizer,accepted,config,round_index)
        state['completed_rounds']+=1;state['gradient_updates']+=metrics['updates']
        state['candidate_groups']+=candidates;state['sampled_responses']+=candidates*a.samples
        state['sampled_response_tokens']+=local_tokens;state['accepted_responses']+=a.groups*a.samples
        state['trained_response_tokens']+=metrics['response_tokens'];state['trained_input_tokens']+=metrics['input_tokens']
        record={'round':state['completed_rounds'],'utc':stamp(),'lr':rate,'candidate_groups':candidates,
            'effective_group_fraction':a.groups/candidates,'counts':dict(counts),'rollout_seconds':rollout_seconds,
            'update_seconds':time.perf_counter()-update_start,'sampled_response_tokens':local_tokens,
            'cumulative_sampled_response_tokens':state['sampled_response_tokens'],
            'cumulative_trained_response_tokens':state['trained_response_tokens'],
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,**metrics}
        append(a.output/'training.jsonl',record);print(json.dumps({'event':'round_complete',**record}),flush=True)
        step=state['completed_rounds']
        if step%config['validation_every']==0 or step==a.rounds:
            result=validate(model,tokenizer,validation,config,a.output/f'validation_round{step:03d}.jsonl')
            state['validation_history'].append({'round':step,**result})
            if result['accuracy']>state['best_validation_accuracy']:
                state['best_validation_accuracy']=result['accuracy'];state['best_round']=step
                save_checkpoint(a.output/'best_checkpoint',model,tokenizer,optimizer,state)
            print(json.dumps({'event':'validation','round':step,**result}),flush=True)
        if step%config['save_every']==0 or step==a.rounds:
            save_checkpoint(a.output/'latest_checkpoint',model,tokenizer,optimizer,state)
        write(a.output/'progress.json',{'status':'running','utc':stamp(),'completed_rounds':step,
            'target_rounds':a.rounds,'gradient_updates':state['gradient_updates'],
            'accepted_responses':state['accepted_responses'],'sampled_responses':state['sampled_responses'],
            'sampled_response_tokens':state['sampled_response_tokens'],'trained_response_tokens':state['trained_response_tokens'],
            'best_validation_accuracy':state['best_validation_accuracy'],'best_round':state['best_round']})
    save_checkpoint(a.output/'final_checkpoint',model,tokenizer,optimizer,state)
    verification=checkpoint_check(a.output/'final_checkpoint',model,optimizer)
    write(a.output/'checkpoint_verification.json',verification)
    for name,digest in config['source_weights_sha256'].items():assert sha(a.checkpoint/name)==digest
    assert sha(a.checkpoint/'checkpoint_manifest.json')==config['source_checkpoint_manifest_sha256']
    result={'status':'complete_verified','finished_utc':stamp(),'completed_rounds':state['completed_rounds'],
        'gradient_updates':state['gradient_updates'],'accepted_responses':state['accepted_responses'],
        'sampled_responses':state['sampled_responses'],'sampled_response_tokens':state['sampled_response_tokens'],
        'trained_response_tokens':state['trained_response_tokens'],'trained_input_tokens':state['trained_input_tokens'],
        'best_validation_accuracy':state['best_validation_accuracy'],'best_round':state['best_round'],
        'validation_history':state['validation_history'],'training_seconds':time.perf_counter()-start,
        'source_sft_unchanged':True,'diagnostic_smoke':a.smoke,'verification':verification}
    result['execution_changes']=state.get('execution_changes',[])
    write(a.output/'result.json',result);write(a.output/'progress.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
