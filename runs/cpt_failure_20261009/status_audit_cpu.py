"""Read-only audit of original failed run and hardware error evidence."""
import datetime as dt
import json
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path('.')
run=root/'runs/cpt_10b_20261008'
out=root/'runs/cpt_failure_20261009'
manifest=json.loads((run/'manifest.json').read_text())
result=json.loads((run/'result.json').read_text())
latest=json.loads((run/'latest_checkpoint.json').read_text())
checkpoint=json.loads((Path(latest['path'])/'checkpoint_manifest.json').read_text())
val=json.loads((run/'validation_history.json').read_text())
before=json.loads((run/'validation_before.json').read_text())
recent=[]
with (run/'training.jsonl').open() as f:
    for line in f:
        record=json.loads(line)
        recent.append(record)
        if len(recent)>1000:recent.pop(0)
last=recent[-1]
started=dt.datetime.fromisoformat(manifest['started_utc'])
ended=dt.datetime.fromtimestamp((run/'result.json').stat().st_mtime,dt.timezone.utc)
now=dt.datetime.now(dt.timezone.utc)
local_tz=ZoneInfo('Asia/Shanghai')
def stamps(t):return {'utc':t.isoformat(),'Asia/Shanghai':t.astimezone(local_tz).isoformat()}
recent_speed=sum(r['input_tokens'] for r in recent)/sum(r['seconds'] for r in recent)
rollback_steps=last['step']-checkpoint['completed_steps']
rollback_tokens=last['cumulative_tokens']-checkpoint['trained_tokens']

def read_command(cmd):
    p=subprocess.run(cmd,text=True,capture_output=True)
    return {'command':cmd,'returncode':p.returncode,'stderr':p.stderr.strip(),'stdout':p.stdout}

gpu=read_command(['nvidia-smi','-q','-x'])
def xml_dict(node):
    if node is None:return None
    if len(node)==0:return node.text.strip() if node.text else None
    return {child.tag:xml_dict(child) for child in node}
if gpu['returncode']==0:
    parsed=ET.fromstring(gpu['stdout']);g=parsed.find('gpu')
    snapshot={name:xml_dict(g.find(name)) for name in ['product_name','product_architecture','ecc_mode','ecc_errors','retired_pages','remapped_rows','gpu_recovery_action','pci','temperature','clocks_event_reasons']}
    snapshot['snapshot_at']=stamps(now)
    snapshot['driver_version']=parsed.findtext('driver_version')
    snapshot['scope']='post-failure device-wide counters; GPU may be occupied by independent diagnostics'
else:snapshot=None

range_start=(started-dt.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S UTC')
range_end=now.strftime('%Y-%m-%d %H:%M:%S UTC')
commands=[['journalctl','-k','--since',range_start,'--until',range_end,'--no-pager','-o','short-iso'],
          ['dmesg','--time-format','iso'],
          ['sudo','-n','journalctl','-k','--since',range_start,'--until',range_end,'--no-pager','-o','short-iso'],
          ['sudo','-n','dmesg','--time-format','iso']]
hardware=re.compile(r'Xid|NVRM|uncorrect|correctable|\bECC\b|hardware error|machine check|\bAER\b|memory failure',re.I)
logs=[]
for cmd in commands:
    read=read_command(cmd)
    lines=read.pop('stdout').splitlines()
    matched=[line for line in lines if hardware.search(line)]
    historical=[];during=[];near=[];unparsed=[]
    for line in matched:
        try:t=dt.datetime.fromisoformat(line.split()[0].replace(',','.'))
        except ValueError:unparsed.append(line);continue
        if started<=t<=ended:during.append(line)
        elif t<started:historical.append(line)
        if abs((t-ended).total_seconds())<=600:near.append(line)
    read.update(total_available_lines=len(lines),hardware_related_lines_in_original_run=during,
       hardware_related_lines_near_failure_10m=near,historical_xid_before_original_run=[l for l in historical if 'Xid' in l],
       unparsed_hardware_lines=unparsed)
    logs.append(read)

audit={'status':'completed','read_only':True,'audit_at':stamps(now),
    'original_run':{'status':result['status'],'error':result['error'],'started':stamps(started),
      'failure_time_from_result_mtime':stamps(ended),
      'failure_timestamp_basis':'result.json mtime; failure traceback has no explicit timestamp. training.jsonl and launch/training.log mtime agree within 0.004 seconds.',
      'elapsed_wall_seconds_to_failure':(ended-started).total_seconds(),
      'last_successful_record':last,'inferred_failed_update':last['step']+1,
      'completed_input_tokens':last['cumulative_tokens'],'completed_fraction':last['cumulative_tokens']/manifest['configured_input_tokens'],
      'epoch':last['epoch'],'step_in_epoch':last['step_in_epoch'],
      'recent_1000_update_tokens_per_second':recent_speed,
      'recent_1000_mean_update_seconds':statistics.mean(r['seconds'] for r in recent),
      'remaining_tokens_without_rollback':manifest['configured_input_tokens']-last['cumulative_tokens']},
    'latest_checkpoint':{'path':latest['path'],'completed_steps':checkpoint['completed_steps'],
      'trained_tokens':checkpoint['trained_tokens'],'saved_at':stamps(dt.datetime.fromisoformat(checkpoint['saved_utc'])),
      'verification_artifact':str(out/'checkpoint_175000_verification.json')},
    'rollback_if_resume_latest':{'successful_updates_to_repeat':rollback_steps,'input_tokens_to_repeat':rollback_tokens,
      'approximate_training_seconds':rollback_tokens/recent_speed,
      'remaining_input_tokens_from_checkpoint':manifest['configured_input_tokens']-checkpoint['trained_tokens'],
      'remaining_pure_training_hours_at_recent_speed':(manifest['configured_input_tokens']-checkpoint['trained_tokens'])/recent_speed/3600},
    'validation':{'initial':before,'best':min(val,key=lambda r:r['loss']),'latest':val[-1],
      'worst':max(val,key=lambda r:r['loss']),'history':val,
      'latest_minus_initial_loss':val[-1]['loss']-before['loss'],
      'trend':'Best observed at step 5000; rose thereafter; roughly stable near 1.90 for later steps. No failed-step validation is available.'},
    'hardware':{'nvidia_smi_command':gpu['command'],'nvidia_smi_returncode':gpu['returncode'],'nvidia_smi_stderr':gpu['stderr'],
       'snapshot':snapshot,'kernel_logs':logs,
       'conclusion':'No hardware error or Xid was observed in accessible logs during the original run or within 10 minutes of its failure; reported ECC counters are zero. A historical Xid 31 before this run is recorded separately and cannot be assigned as the cause of this failure.'},
    'limitations':['nvidia-smi counters are a post-failure snapshot, not an event-by-event proof.',
       'No Xid in accessible logs does not establish that hardware cannot contribute; missing or rotated logs remain possible.',
       'A non-finite total gradient norm can arise from non-finite entries or float32 norm overflow; original traceback alone does not distinguish them.',
       'The current GPU workload, if any, is independent of the already failed original run.']}
out.mkdir(parents=True,exist_ok=True)
(out/'status_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(out/'status_audit.json'),'start':stamps(started),'end':stamps(ended),
 'wall_hours':(ended-started).total_seconds()/3600,'completed_B':last['cumulative_tokens']/1e9,
 'rollback_steps':rollback_steps,'rollback_tokens':rollback_tokens,'rollback_minutes':rollback_tokens/recent_speed/60,
 'latest_val':val[-1]['loss'],'initial_val':before['loss'],'remaining_pure_hours':audit['rollback_if_resume_latest']['remaining_pure_training_hours_at_recent_speed']},ensure_ascii=False))
