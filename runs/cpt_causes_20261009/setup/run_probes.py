"""Evaluate fixed samples at three immutable weight snapshots, sequentially."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

root = Path('.')
diag = root / 'runs/cpt_causes_20261009'
setup = diag / 'setup'
original = root / 'runs/cpt_native_from0_5h_20261009'
history = json.loads((original / 'validation_history.json').read_text())
expected = {row['step']: row['loss'] for row in history}
base = json.loads((original / 'validation_before.json').read_text())['loss']
cases = [
    ('base', 'Qwen/Qwen2.5-Coder-1.5B', base, 'df3ce67c0e24480f20468b6ef2894622d69eb73b'),
    ('step20000', str(original / 'checkpoint_step_00020000'), expected[20000], None),
    ('step30860', str(original / 'stopped_checkpoint'), expected[30860], None),
]
result = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
          'training_updates': 0, 'production_files_modified': False, 'cases': {}}
summary = diag / 'probe_evaluation_summary.json'

def save():
    temp = summary.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(summary)

save()
for name, model, loss, revision in cases:
    command = [sys.executable, str(setup / 'tools/evaluate_cpt_seen_unseen.py'),
               '--source', str(setup / 'source'), '--data', str(root / 'data/processed/openwebmath_5b_s42'),
               '--probes', str(diag / 'probes.json'), '--model', model, '--label', name,
               '--expected-validation', str(loss), '--output', str(diag / ('probe_eval_' + name))]
    if revision:
        command += ['--revision', revision]
    with (setup / ('probe_eval_' + name + '.log')).open('w') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        result.update(status='failed', failed_case=name, exit_code=process.returncode)
        save()
        sys.exit(process.returncode)
    row = json.loads((diag / ('probe_eval_' + name) / 'result.json').read_text())
    result['cases'][name] = {'metrics': row['metrics'], 'probes_sha256': row['probes_sha256']}
    save()
result.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
base_metrics = result['cases']['base']['metrics']
result['loss_changes_from_base'] = {
    name: {group: metric['loss'] - base_metrics[group]['loss'] for group, metric in row['metrics'].items()}
    for name, row in result['cases'].items() if name != 'base'
}
result['limit'] = 'These fixed samples distinguish previously seen and unseen blocks. Three observational snapshots do not establish a cause or exhaustive absence of memorization.'
save()
