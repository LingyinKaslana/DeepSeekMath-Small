"""Native HF answer-only SFT; FP32 masters/Adam, BF16 autocast, no packing."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from standard_native import update_native, preserve_rng


def write(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


class Examples:
    def __init__(self,path,eos,device='cuda'):
        with path.open() as stream:
            self.rows=[json.loads(line) for line in stream]
        self.eos=eos;self.device=device
        for row in self.rows:
            assert 1<row['prompt_length']<len(row['input_ids'])<=4096
            assert row['input_ids'][-1]==eos and eos not in row['input_ids'][:-1]

    def __len__(self):return len(self.rows)


def batch(dataset,indices):
    rows=[dataset.rows[int(index)] for index in indices]
    width=math.ceil(max(len(row['input_ids']) for row in rows)/8)*8
    ids=torch.full((len(rows),width),dataset.eos,dtype=torch.long)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for index,row in enumerate(rows):
        length=len(row['input_ids']);prompt=row['prompt_length']
        ids[index,:length]=torch.tensor(row['input_ids'])
        labels[index,prompt:length]=ids[index,prompt:length]
        mask[index,:length]=1
    count=int((labels[:,1:]!=-100).sum());inputs=int(mask.sum())
    assert count==sum(len(row['input_ids'])-row['prompt_length'] for row in rows)
    return {key:value.to(dataset.device) for key,value in
            {'input_ids':ids,'attention_mask':mask,'labels':labels}.items()},count,inputs


@torch.inference_mode()
def evaluate(model,dataset,indices,micro=8):
    was_training=model.training;model.eval()
    ordered=sorted(indices,key=lambda i:len(dataset.rows[i]['input_ids']))
    total=torch.zeros((),device='cuda');targets=inputs=0
    with preserve_rng(torch.device('cuda')):
        for offset in range(0,len(ordered),micro):
            values,count,n=batch(dataset,ordered[offset:offset+micro])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**values,use_cache=False).loss
            total+=loss*count;targets+=count;inputs+=n
    value=float(total/targets);assert math.isfinite(value)
    model.train(was_training)
    return {'loss':value,'supervised_tokens':targets,'input_tokens':inputs,'examples':len(indices),
            'loss_scope':'Answer tokens and EOS only; prompt and padding ignored'}


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu(item) for item in value)
    return value


def save(path,model,tokenizer,optimizer,state):
    temporary=path.with_name(path.name+'.incomplete')
    assert not temporary.exists()
    temporary.mkdir()
    model.save_pretrained(temporary,safe_serialization=True,max_shard_size='4GB')
    tokenizer.save_pretrained(temporary)
    payload={**state,'optimizer':cpu(optimizer.state_dict()),'cpu_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state()}
    torch.save(payload,temporary/'training_state.pt')
    metadata={key:value for key,value in state.items() if key not in ('invocations',)}
    metadata.update(stage='sft',parameter_dtype='float32',optimizer_dtype='float32')
    write(temporary/'checkpoint_manifest.json',metadata)
    backup=path.with_name(path.name+'.previous')
    assert not backup.exists()
    if path.exists():path.rename(backup)
    temporary.rename(path)
    if backup.exists():shutil.rmtree(backup)
    del payload;gc.collect()


@torch.inference_mode()
def sample(model,tokenizer,dataset,path):
    model.eval();result=[]
    indices=[next(i for i,r in enumerate(dataset.rows) if r['dataset']==name) for name in ('gsm8k','math')]
    with preserve_rng(torch.device('cuda')),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in indices:
            row=dataset.rows[i];ids=torch.tensor([row['input_ids'][:row['prompt_length']]],device='cuda')
            config=GenerationConfig(do_sample=False,max_new_tokens=256,use_cache=True,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id)
            generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                                     generation_config=config,logits_to_keep=1)
            result.append({'dataset':row['dataset'],'source_id':row['source_id'],'prompt':row['prompt'],
                'completion':tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),
                'reference':row['completion']})
    write(path,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--micro-batch',type=int,default=8)
    p.add_argument('--effective-batch',type=int,default=32)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--stress-longest',action='store_true')
    p.add_argument('--resume',type=Path)
    a=p.parse_args();assert a.epochs>0 and a.lr>0
    assert not a.stress_longest or a.stop_after is not None
    assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    data=json.loads((a.data/'manifest.json').read_text());assert data['status']=='ready'
    for name,info in data['artifacts'].items():assert digest(a.data/name)==info['sha256']
    parent=json.loads((a.checkpoint/'checkpoint_manifest.json').read_text())
    assert parent['trained_tokens']==10000000000 and parent['completed_steps']==305176
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    train=Examples(a.data/'train.jsonl',tokenizer.eos_token_id)
    validation=Examples(a.data/'validation.jsonl',tokenizer.eos_token_id)
    steps_per_epoch=math.ceil(len(train)/a.effective_batch);total=steps_per_epoch*a.epochs
    warmup=max(1,math.ceil(total*.03))
    config={'epochs':a.epochs,'lr':a.lr,'min_lr':a.lr*.1,'warmup_steps':warmup,
        'total_steps':total,'micro_batch':a.micro_batch,'effective_batch':a.effective_batch,
        'seed':42,'weight_decay':0.0,'betas':[.9,.95],'max_sequence_length':4096,
        'source_checkpoint':str(a.checkpoint.resolve()),'source_cpt_input_tokens':10000000000,
        'source_checkpoint_manifest_sha256':digest(a.checkpoint/'checkpoint_manifest.json'),
        'source_weights_sha256':{path.name:digest(path) for path in sorted(a.checkpoint.glob('*.safetensors'))},
        'data_manifest_sha256':digest(a.data/'manifest.json'),
        'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in
                         ('train_math_sft.py','standard_native.py')}}
    manifest={'stage':'sft','started_utc':datetime.now(timezone.utc).isoformat(),'config':config,
              'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()},
              'data':data,'optimizer_initialization':'Fresh AdamW for SFT; CPT optimizer not loaded',
              'precision':'FP32 parameters/gradients/Adam; BF16 autocast; native HF SDPA'}
    write(a.output/'manifest.json',manifest)
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(a.resume or a.checkpoint,local_files_only=True,
             torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda')
    model.config.use_cache=False
    assert all(parameter.dtype==torch.float32 and parameter.requires_grad for parameter in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),eps=1e-8,
                                weight_decay=0.,foreach=True,fused=False)
    state={'stage':'sft','config':config,'completed_steps':0,'sft_input_tokens':0,
        'sft_supervised_tokens':0,'source_cpt_input_tokens':10000000000,'validation_history':[],
        'train_probe_history':[],'best_validation_loss':float('inf'),'best_step':None,
        'diagnostic_smoke':a.stop_after is not None}
    if a.resume:
        restored=torch.load(a.resume/'training_state.pt',map_location='cpu',weights_only=True)
        assert restored['config']==config
        assert not restored.get('diagnostic_smoke'), 'Diagnostic smoke cannot seed formal SFT'
        optimizer.load_state_dict(restored.pop('optimizer'))
        torch.set_rng_state(restored.pop('cpu_rng'));torch.cuda.set_rng_state(restored.pop('cuda_rng'))
        state=restored
    else:assert not optimizer.state
    print(json.dumps({'event':'model_loaded','seconds':time.perf_counter()-started,
          'fresh_optimizer':a.resume is None,'parameters':sum(p.numel() for p in model.parameters()),
          'config':config}),flush=True)
    probe=random.Random(43).sample(range(len(train)),128)
    before=evaluate(model,validation,list(range(len(validation))))
    write(a.output/'validation_before.json',before)
    print(json.dumps({'event':'validation_before','step':state['completed_steps'],**before}),flush=True)
    if not a.resume:
        state['best_validation_loss']=before['loss']
        state['validation_history'].append({'step':0,**before})
        state['train_probe_history'].append({'step':0,**evaluate(model,train,probe)})
    torch.cuda.reset_peak_memory_stats();orders={}
    with (a.output/'training.jsonl').open('w') as log:
        while state['completed_steps']<total:
            if a.stop_after is not None and state['completed_steps']>=a.stop_after:break
            step=state['completed_steps'];epoch,local=divmod(step,steps_per_epoch)
            if epoch not in orders:
                order=list(range(len(train)));random.Random(42+epoch).shuffle(order);orders[epoch]=order
            indices=orders[epoch][local*a.effective_batch:(local+1)*a.effective_batch]
            stress=bool(a.stress_longest and step==0)
            if stress:indices=sorted(range(len(train)),key=lambda i:len(train.rows[i]['input_ids']),reverse=True)[:a.effective_batch]
            indices=sorted(indices,key=lambda i:len(train.rows[i]['input_ids']))
            rate=a.lr*(step+1)/warmup if step<warmup else a.lr*(.1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-1-warmup))))
            for group in optimizer.param_groups:group['lr']=rate
            begin=time.perf_counter()
            loss,norm,inputs,targets,applied=update_native(model,optimizer,train,batch,indices,a.micro_batch)
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            assert applied and math.isfinite(float(loss)) and math.isfinite(float(norm))
            state['completed_steps']+=1;state['sft_input_tokens']+=inputs;state['sft_supervised_tokens']+=targets
            row={'step':state['completed_steps'],'epoch':epoch+1,'loss':float(loss),'grad_norm':float(norm),
                'lr':rate,'input_tokens':inputs,'supervised_tokens':targets,'seconds':seconds,
                'cumulative_sft_input_tokens':state['sft_input_tokens'],'stress_longest':stress,
                'optimizer_applied':applied,'max_sequence_in_batch':max(len(train.rows[i]['input_ids']) for i in indices),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
            log.write(json.dumps(row)+'\n');log.flush()
            if state['completed_steps']%10==0 or stress:
                write(a.output/'progress.json',row);print(json.dumps({'event':'training',**row}),flush=True)
            finished=state['completed_steps']==total
            stopping=a.stop_after is not None and state['completed_steps']>=a.stop_after
            if state['completed_steps']%a.eval_every==0 or state['completed_steps']%steps_per_epoch==0 or finished or stopping:
                metrics={'step':state['completed_steps'],**evaluate(model,validation,list(range(len(validation))))}
                probe_metrics={'step':state['completed_steps'],**evaluate(model,train,probe)}
                state['validation_history'].append(metrics);state['train_probe_history'].append(probe_metrics)
                improved=metrics['loss']<state['best_validation_loss']
                if improved:state['best_validation_loss']=metrics['loss'];state['best_step']=state['completed_steps']
                write(a.output/'validation_history.json',state['validation_history'])
                write(a.output/'train_probe_history.json',state['train_probe_history'])
                print(json.dumps({'event':'validation',**metrics,'best_step':state['best_step']}),flush=True)
                if improved and a.stop_after is None:save(a.output/'best_checkpoint',model,tokenizer,optimizer,state)
            if state['completed_steps']%steps_per_epoch==0 and not finished and a.stop_after is None:
                save(a.output/'latest_checkpoint',model,tokenizer,optimizer,state)
    path=a.output/('stopped_checkpoint' if a.stop_after else 'final_checkpoint')
    save(path,model,tokenizer,optimizer,state)
    sample(model,tokenizer,validation,a.output/'generation_samples.json')
    result={'status':'stopped' if a.stop_after else 'complete','stage':'sft',
        'completed_steps':state['completed_steps'],'configured_steps':total,
        'sft_input_tokens':state['sft_input_tokens'],'sft_supervised_tokens':state['sft_supervised_tokens'],
        'validation_before':before,'validation_after':state['validation_history'][-1],
        'best_step':state['best_step'],'best_validation_loss':state['best_validation_loss'],
        'final_checkpoint':str(path.resolve()),'evaluation_checkpoint':str((a.output/'best_checkpoint').resolve()) if state['best_step'] is not None and a.stop_after is None else str(path.resolve()),
        'wall_seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'cpt_checkpoint_modified':False}
    write(a.output/'result.json',result);write(a.output/'progress.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
