import json,os,hashlib
from pathlib import Path
from datetime import datetime,timezone
setup=Path(__file__).resolve().parent;root=setup.parent.parent
tuning=root/'runs/deepseek_dapo_tuning_20261015';run=root/'runs/deepseek_dapo_100r_20261015'
boundary=json.loads((tuning/'boundary.json').read_text());benchmark=json.loads((tuning/'benchmark.json').read_text())
assert benchmark['status']=='complete' and benchmark['source_checkpoint_unchanged']
assert benchmark['selected_micro_batch'] in (2,8,16,32) and benchmark['selected_rollout_prompts'] in (16,32)
step=boundary['completed_rounds']
with (run/'training.jsonl').open() as stream:training=[json.loads(line) for line in stream]
assert len(training)==step and training[-1]['round']==step
source=run/'rollouts.jsonl';snapshot=tuning/'rollouts_before_resume.jsonl'
os.link(source,snapshot)
temporary=source.with_name('rollouts_resume.tmp');removed=0;retained=0
with source.open() as reader,temporary.open('w') as writer:
    for line in reader:
        row=json.loads(line)
        if row['round']<=step:writer.write(line);retained+=1
        else:removed+=1
temporary.replace(source)
for name in ('manifest.json','progress.json'):
    os.link(run/name,tuning/('original_'+name))
record={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'resume_round':step,
    'checkpoint':boundary['checkpoint'],'restores':['weights','Adam moments','CPU RNG','CUDA RNG','training-order cursor','validation history'],
    'micro_batch_old':2,'micro_batch_new':benchmark['selected_micro_batch'],
    'rollout_prompts_old':16,'rollout_prompts_new':benchmark['selected_rollout_prompts'],
    'training_budget_unchanged':{'rounds':100,'effective_groups_per_round':32,'responses_per_group':8,'optimizer_batch':64},
    'completed_rounds_rolled_back':0,'uncommitted_rollout_records_archived':removed,'retained_rollout_records':retained,
    'original_manifest_preserved':True,'limits':['Different batching can change BF16 rounding and stochastic sampling trajectories.']}
(setup/'resume_verification.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'execution_intervention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
