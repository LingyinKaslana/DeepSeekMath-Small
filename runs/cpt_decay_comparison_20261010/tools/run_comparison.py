"""Sequential bounded branches and automatic report; production stays running."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

DIAG = Path(__file__).resolve().parents[1]
ROOT = DIAG.parents[1]
PLAN = json.loads((DIAG/'plan.json').read_text())
STATUS = dict(status='starting',started_utc=datetime.now(timezone.utc).isoformat(),
    coordinator_pid=os.getpid(),production_pid=PLAN['production_pid'],production_files_modified=False,
    active_case=None,branch_pid=None,cases={},report=None)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save():
    path=DIAG/'suite_status.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(STATUS,indent=2)+'\n')
    temporary.replace(path)

def main():
    review=json.loads((DIAG/'preflight_independent_review.json').read_text())
    if (review.get('status')!='passed_static_cpu'
        or review.get('checks_passed')!=review.get('checks_total')
        or review.get('blocking_findings')):
        raise RuntimeError('Independent preflight did not pass')
    if sha(DIAG/'tools/train_branch.py')!=review['runner_sha256'] or sha(DIAG/'plan.json')!=review['experiment_plan_sha256']:
        raise RuntimeError('Reviewed runner or experiment plan changed')
    if not json.loads((DIAG/'paired_cpu_preflight.json').read_text()).get('passed'):
        raise RuntimeError('Paired CPU preflight did not pass')
    runtime_tools={name:sha(DIAG/'tools'/name) for name in
        ('train_branch.py','compare_cpt_backends.py','compare_cpt_hyperparameters.py','summarize_comparison.py','run_comparison.py')}
    STATUS['runtime_tools_sha256']=runtime_tools
    if not Path('/proc',str(PLAN['production_pid'])).exists():
        raise RuntimeError('Expected production process missing')
    STATUS['production_proc_starttime']=Path('/proc',str(PLAN['production_pid']),'stat').read_text().split()[21]
    save()
    for arm in PLAN['arms']:
        case=arm['id']
        STATUS.update(status='running',active_case=case)
        save()
        command=[sys.executable,str(DIAG/'tools/train_branch.py'),'--case',case]
        with (DIAG/('branch_'+case+'.log')).open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            STATUS['branch_pid']=process.pid
            save()
            code=process.wait()
        if code:
            raise RuntimeError(f'Branch {case} failed with exit{code}; see branch log')
        result_path=Path(arm['run_directory'])/'result.json'
        result=json.loads(result_path.read_text())
        if result['status']!='complete' or result['optimizer_updates_applied']!=PLAN['steps']:
            raise RuntimeError('Branch did not finish its exact diagnostic budget')
        STATUS['cases'][case]=dict(result=str(result_path),end_step=result['end_step'],
            metrics={name:row['loss'] for name,row in result['evaluation_history'][-1].items() if isinstance(row,dict) and 'loss' in row})
        STATUS['branch_pid']=None
        save()
    STATUS.update(status='generating_report',active_case=None)
    save()
    command=[str(ROOT/'.venv-plot/bin/python'),str(DIAG/'tools/summarize_comparison.py'),'--experiment',str(DIAG)]
    with (DIAG/'report.log').open('x') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('Report generation failed')
    if Path('/proc',str(PLAN['production_pid'])).exists():
        actual=Path('/proc',str(PLAN['production_pid']),'stat').read_text().split()[21]
        if actual!=STATUS['production_proc_starttime']:
            raise RuntimeError('Production process identity changed during comparison')
        STATUS['production_same_process_still_running']=True
    else:
        STATUS['production_same_process_still_running']=False
        STATUS['production_status_note']='Production may have completed independently; diagnostic never sends signals.'
    STATUS.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),report=str(DIAG/'report'))
    save()

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        STATUS.update(status='failed',error=str(error),traceback=traceback.format_exc(),
            finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
