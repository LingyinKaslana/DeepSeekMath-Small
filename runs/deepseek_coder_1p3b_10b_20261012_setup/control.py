import argparse,json
from datetime import datetime,timezone
from pathlib import Path
root=Path('.')
setup=root/'runs/deepseek_coder_1p3b_10b_20261012_setup'
run=root/'runs/deepseek_coder_1p3b_10b_20261012'
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--status',required=True);p.add_argument('--exit-code',type=int);a=p.parse_args()
now=datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def write(path,data):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
status={'status':a.status,'phase':a.phase,'updated_utc':now,'run':str(run),'source_run':str(root/'runs/deepseek_coder_1p3b_500m_20261012'),'model':'deepseek-ai/deepseek-coder-1.3b-base','configured_input_tokens':10000000000,'starting_input_tokens':500006912,'optimizer_preserved':True,'warmup_repeated':False,'qwen_experiments_modified':False,'old_pilot_modified':False}
if a.exit_code is not None:status['exit_code']=a.exit_code
if a.status=='complete_verified':
 result=read(run/'result.json');review=read(setup/'final_checkpoint_verification.json');plot=read(setup/'report/summary.json')
 assert result['status']=='complete' and result['trained_tokens']==10000000000 and result['completed_steps']==305176
 assert review['status']=='passed' and plot['common_evaluation_step']==305176 and plot['latest']['input_tokens']==10000000000
 status.update(result=str(run/'result.json'),checkpoint=result['checkpoint'],verification=str(setup/'final_checkpoint_verification.json'),report=str(setup/'report/summary.json'),curve=str(setup/'report/train_validation_curve.png'))
 write(setup/'completion.json',{**status,'training':result,'checkpoint_review':review,'curves':plot})
write(setup/'status.json',status)
for name in ('plan.json','launch.json'):
 path=setup/name
 if path.exists():
  metadata=read(path);metadata.update(status=a.status,phase=a.phase,updated_utc=now)
  if a.status=='complete_verified':metadata['completed_utc']=now
  if a.exit_code is not None:metadata['exit_code']=a.exit_code
  write(path,metadata)
active={**status,'setup':str(setup),'progress':str(run/'progress.json'),'latest_checkpoint':str(run/'latest_checkpoint.json')}
for path in (root/'runs/deepseek_coder_active_run.json',root/'runs/cpt_active_run.json'):write(path,active)
print(json.dumps(status,ensure_ascii=False),flush=True)
