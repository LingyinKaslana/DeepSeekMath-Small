"""Record the actual bounded experiment outcome after the wrapper exits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--setup', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--stage', required=True)
p.add_argument('--exit-code', type=int, required=True)
a = p.parse_args()
result_path = a.run / 'result.json'
r = json.loads(result_path.read_text()) if result_path.exists() else {}
record = {
    'updated_utc': datetime.now(timezone.utc).isoformat(),
    'stage': a.stage,
    'exit_code': a.exit_code,
    'status': ('completed_observation' if a.stage == 'finished'
               and a.exit_code == 0 and r.get('stop_reason') == 'max_seconds'
               else 'stopped_early' if a.stage == 'finished' and a.exit_code == 0
               else 'failed'),
    'run': str(a.run),
    'training_status': r.get('status'),
    'stop_reason': r.get('stop_reason'),
    'completed_steps': r.get('completed_steps'),
    'trained_tokens': r.get('trained_tokens'),
    'configured_reference_tokens': 10_000_000_000,
    'production_budget_complete': False,
    'old_production_resumed': False,
}
if a.stage == 'finished' and a.exit_code == 0:
    record['checkpoint_verification'] = str(a.setup / 'final_checkpoint_verification.json')
    record['report'] = str(a.setup / 'report')
(a.setup / 'observation_status.json').write_text(json.dumps(record, indent=2) + '\n')
