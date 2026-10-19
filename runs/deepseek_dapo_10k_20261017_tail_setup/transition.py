"""Switch at the next complete checkpoint; preserve and audit the old experiment."""
import os,signal,subprocess,time,json,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
setup=Path(__file__).resolve().parent;root=setup.parent.parent
old=root/'runs/deepseek_dapo_10k_20261016_setup';run=root/'runs/deepseek_dapo_10k_20261016'
bench=root/'runs/deepseek_dapo_tailopt_20261017';pid=1936427

def write(path,record):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
def status(phase,**kwargs):
 record={'status':phase,'utc':datetime.now(timezone.utc).isoformat(),**kwargs}
 write(setup/'transition_status.json',record);print(json.dumps(record),flush=True)
def last_training_round():
 with (run/'training.jsonl').open('rb') as stream:
  stream.seek(max(0,stream.seek(0,2)-32768));lines=stream.read().splitlines()
 return json.loads(lines[-1])['round']
def alive():
 try:return 'dapo_math.py' in Path(f'/proc/{pid}/cmdline').read_text()
 except FileNotFoundError:return False

# First require successful small-concurrency numerical and update tests.
while True:
 if (bench/'main_pause.json').exists() and 'resumed_unix' in json.loads((bench/'main_pause.json').read_text()):
  result=json.loads((bench/'benchmark.json').read_text()) if (bench/'benchmark.json').exists() else {}
  assert result.get('status')=='complete','Small-concurrency experiment did not pass; main training remains unchanged'
  break
 status('waiting_for_isolated_benchmark');time.sleep(10)
status('waiting_for_checkpoint',minimum_round=2000)
while True:
 try:manifest=json.loads((run/'latest_checkpoint/checkpoint_manifest.json').read_text())
 except FileNotFoundError:time.sleep(.2);continue
 if manifest['completed_rounds']>=2000 and alive() and (run/'checkpoints'/f"round{manifest['completed_rounds']:06d}"/'checkpoint_manifest.json').exists():
  os.kill(pid,signal.SIGSTOP);time.sleep(.2)
  try:
   manifest=json.loads((run/'latest_checkpoint/checkpoint_manifest.json').read_text())
   if last_training_round()==manifest['completed_rounds']:break
  except BaseException:
   os.kill(pid,signal.SIGCONT);raise
  os.kill(pid,signal.SIGCONT)
 if not alive():
  # Preserve explicit failure evidence; recover from the most recent complete state.
  manifest=json.loads((run/'latest_checkpoint/checkpoint_manifest.json').read_text());break
 time.sleep(.2)
step=manifest['completed_rounds'];last_round=last_training_round();old_alive=alive()
parent_pid=int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('PPid:'))) if old_alive else None
try:
 for name,digest in json.loads((setup/'source_hashes.json').read_text()).items():
  with (setup/'source'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
 assert (run/'latest_checkpoint/training_state.pt').exists()
 # Atomic save uses replacement directories: hardlinks preserve these immutable files.
 shutil.copytree(run/'latest_checkpoint',setup/'resume_checkpoint',copy_function=os.link)
 record={'status':'checkpoint_acquired','utc':datetime.now(timezone.utc).isoformat(),'completed_rounds':step,
         'last_observed_completed_round':last_round,'completed_round_rollback':last_round-step,
         'abandoned_completed_groups_used':(last_round-step)*32,'abandoned_gradient_updates':(last_round-step)*4,
         'recovery_reason':None if last_round==step else 'Original process ended before next checkpoint; abandoned branch archived',
         'checkpoint':str(setup/'resume_checkpoint'),'gradient_updates':manifest['gradient_updates'],
         'effective_groups_used':manifest['effective_groups_used'],'effective_groups_found':manifest['effective_groups_found']}
 write(setup/'boundary.json',record)
 # Preserve all raw logs before removing uncommitted (or failed-branch) records.
 prepared_logs={}
 for name in ('training.jsonl','rollouts.jsonl'):
  source=run/name;os.link(source,setup/('before_transition_'+name));temporary=source.with_suffix('.transition.tmp');kept=removed=0
  with source.open() as reader,temporary.open('w') as writer:
   for line in reader:
    try:item=json.loads(line)
    except json.JSONDecodeError:removed+=1;continue
    if item['round']<=step:writer.write(line);kept+=1
    else:removed+=1
  prepared_logs[name]=temporary;record[name]={'kept':kept,'archived_removed':removed}
 with (setup/'history_bootstrap.log').open('w') as log:
  subprocess.run([str(root/'.venv/bin/python'),str(setup/'source/audit_dapo_prompt_history.py'),'--run',str(run),
                  '--output',str(setup/'bootstrap'),'--through-round',str(step)],stdout=log,stderr=subprocess.STDOUT,check=True)
 with prepared_logs['training.jsonl'].open() as stream:training=[json.loads(line) for line in stream]
 assert [r['round'] for r in training]==list(range(1,step+1))
 assert sum(r['candidate_groups'] for r in training)==manifest['candidate_groups']
 assert sum(r['counts']['mixed'] for r in training)==manifest['effective_groups_found']
 assert sum(r['response_tokens'] for r in training)==manifest['trained_response_tokens']
 assert record['rollouts.jsonl']['kept']==manifest['candidate_groups']
 assert manifest['effective_groups_used']==step*32 and manifest['gradient_updates']==step*4
 # Only now release the original GPU process. Its full saved state is preserved.
 if old_alive:
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  for _ in range(300):
   if not alive():break
   time.sleep(.1)
  assert not alive(),'Original process did not release GPU'
 for name,temporary in prepared_logs.items():temporary.replace(run/name)
 if parent_pid is not None:
  for _ in range(300):
   if not Path(f'/proc/{parent_pid}').exists():break
   time.sleep(.1)
 write(old/'optimization_transition.json',record);write(setup/'boundary.json',record)
 old_status=json.loads((old/'status.json').read_text());old_status.update(status='superseded_at_checkpoint',phase='user_authorized_sampling_optimization',transition=str(setup/'boundary.json'))
 write(old/'status.json',old_status)
 status('checkpoint_ready',completed_rounds=step,selected_small_probe=result['selected_policy'])
except BaseException:
 if old_alive and alive():os.kill(pid,signal.SIGCONT)
 raise
os.execv('/bin/bash',['bash',str(setup/'run.sh')])
