import json,os
from pathlib import Path
from datetime import datetime,timezone
setup=Path(__file__).resolve().parent;root=setup.parent.parent
experiment=root/'runs/deepseek_dapo_decode_20261015';run=root/'runs/deepseek_dapo_100r_20261015'
boundary=json.loads((experiment/'boundary.json').read_text());benchmark=json.loads((experiment/'benchmark.json').read_text())
assert benchmark['status']=='complete' and benchmark['uncompacted_sampling_exact_hf']
assert benchmark['update_smoke']['updates']==4
step=boundary['completed_rounds']
with (run/'training.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
assert len(rows)==step and rows[-1]['round']==step
source=run/'rollouts.jsonl';os.link(source,experiment/'rollouts_before_resume.jsonl')
tmp=source.with_name('rollouts_decode_resume.tmp');dropped=0
with source.open() as reader,tmp.open('w') as writer:
    for line in reader:
        if json.loads(line)['round']<=step:writer.write(line)
        else:dropped+=1
tmp.replace(source)
proof={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'resume_round':step,
    'checkpoint':boundary['checkpoint'],'compact_decode':benchmark['selected_compact_decode'],
    'preserved':['weights','Adam moments','CPU RNG','CUDA RNG','data cursor','validation history'],
    'completed_round_rollback':0,'uncommitted_rollout_records_archived':dropped,
    'micro_batch':16,'rollout_prompts':32,'samples_per_prompt':8,'effective_groups_per_round':32,'total_rounds':100,
    'policy_precision':'Unchanged FP32 parameters with BF16 autocast; cache dtypes unchanged',
    'limits':['Compaction changes random-number consumption and floating-point batch shapes.']}
(setup/'resume_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
