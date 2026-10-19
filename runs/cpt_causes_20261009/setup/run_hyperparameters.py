"""Bounded sequential diagnostic branches; the production stays paused."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('.')
diag = root / 'runs/cpt_causes_20261009'
setup = diag / 'setup'
source_run = root / 'runs/cpt_native_from0_5h_20261009'
status_path = diag / 'hyperparameter_suite_status.json'
status = {'status': 'waiting_for_probe_evaluations',
          'started_utc': datetime.now(timezone.utc).isoformat(),
          'production_resumed': False, 'steps_per_branch': 1000, 'cases': {}}

def save():
    temp = status_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(status_path)

save()
# Probe evaluation uses the same GPU. Wait for its verified normal completion.
while True:
    p = diag / 'probe_evaluation_summary.json'
    if p.exists():
        probe_status = json.loads(p.read_text())['status']
        if probe_status == 'complete':
            break
        if probe_status == 'failed':
            status.update(status='blocked_by_probe_evaluation_failure')
            save()
            sys.exit(1)
    time.sleep(5)

for case in ('baseline', 'lr_0p3', 'wd_zero'):
    status.update(status='running', active_case=case)
    save()
    command = [sys.executable, str(setup / 'tools/compare_cpt_hyperparameters.py'),
               '--run', str(source_run), '--checkpoint', str(source_run / 'checkpoint_step_00020000'),
               '--source', str(setup / 'source'), '--output', str(diag / ('branch_' + case)),
               '--case', case, '--steps', '1000', '--eval-every', '500',
               '--probes-json', str(diag / 'probes.json')]
    with (setup / ('branch_' + case + '.log')).open('w') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        status.update(status='failed', failed_case=case, exit_code=process.returncode)
        save()
        sys.exit(process.returncode)
    result = json.loads((diag / ('branch_' + case) / 'result.json').read_text())
    if result['status'] != 'complete':
        raise RuntimeError('Branch did not complete its fixed diagnostic budget')
    status['cases'][case] = {'result': str(diag / ('branch_' + case) / 'result.json'),
                           'start_step': result['start_step'], 'end_step': result['end_step'],
                           'new_input_tokens': result['new_input_tokens']}
    save()

status.update(status='completed_branches', active_case=None,
              finished_training_utc=datetime.now(timezone.utc).isoformat())
save()
command = [str(root / '.venv-plot/bin/python'), str(setup / 'tools/report_cpt_hyperparameters.py'),
           '--directory', str(diag)]
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(setup / 'final_report.log').write_text(process.stdout)
if process.returncode != 0:
    status.update(status='report_failed', exit_code=process.returncode)
    save()
    sys.exit(process.returncode)
status.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
save()
