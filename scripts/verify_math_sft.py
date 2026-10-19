"""Independent CPU reload of full SFT weights, optimizer and token counters."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8);started=time.perf_counter()
    checkpoint=a.checkpoint.resolve()
    meta=json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True,mmap=True)
    assert meta['stage']==state['stage']=='sft'
    for key in ('config','completed_steps','sft_input_tokens','sft_supervised_tokens','source_cpt_input_tokens'):
        assert meta[key]==state[key]
    parent=Path(state['config']['source_checkpoint'])
    assert sha(parent/'checkpoint_manifest.json')==state['config']['source_checkpoint_manifest_sha256']
    for name,digest in state['config']['source_weights_sha256'].items():assert sha(parent/name)==digest
    with (checkpoint.parent/'training.jsonl').open() as stream:
        rows=[json.loads(line) for line in stream]
    # A checkpoint can precede the end of this invocation, including validation-selected best.
    prefix=[row for row in rows if row['step']<=state['completed_steps']]
    if prefix and prefix[0]['step']==1:
        assert len(prefix)==state['completed_steps']
        assert sum(row['input_tokens'] for row in prefix)==state['sft_input_tokens']
        assert sum(row['supervised_tokens'] for row in prefix)==state['sft_supervised_tokens']
    assert state['source_cpt_input_tokens']==10000000000
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,torch_dtype=torch.float32)
    parameters=list(model.parameters());weights=model.state_dict();checked=set()
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in parameters)
    for path in checkpoint.glob('*.safetensors'):
        with safe_open(path,framework='pt',device='cpu') as stream:
            for key in stream.keys():
                assert key not in checked and torch.equal(weights[key],stream.get_tensor(key))
                checked.add(key)
    assert checked==set(weights)
    saved=state['optimizer'];assert len(saved['param_groups'])==1
    group=saved['param_groups'][0]
    assert tuple(group['betas'])==(.9,.95) and group['weight_decay']==0.0 and group['eps']==1e-8
    assert group['params']==list(range(len(parameters)))
    assert len(saved['state'])==len(parameters)
    for index,parameter in enumerate(parameters):
        item=saved['state'][index]
        assert float(item['step'])==state['completed_steps']
        for name in ('exp_avg','exp_avg_sq'):
            tensor=item[name]
            assert tensor.dtype==torch.float32 and tensor.shape==parameter.shape and bool(torch.isfinite(tensor).all())
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=0.,foreach=True,fused=False)
    optimizer.load_state_dict(saved)
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'cpu_only':True,
        'checkpoint':str(checkpoint),'weight_tensors_exactly_reloaded':len(checked),
        'parameters':sum(p.numel() for p in parameters),'optimizer_states':len(saved['state']),
        'completed_steps':state['completed_steps'],'sft_input_tokens':state['sft_input_tokens'],
        'sft_supervised_tokens':state['sft_supervised_tokens'],'source_cpt_unchanged':True,
        'seconds':time.perf_counter()-started}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
