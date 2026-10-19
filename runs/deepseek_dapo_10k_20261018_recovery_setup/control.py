import argparse,json
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--status',required=True);p.add_argument('--exit-code',type=int);a=p.parse_args()
setup=Path(__file__).resolve().parent;run=setup.parent/'deepseek_dapo_10k_20261016'
record={'status':a.status,'phase':a.phase,'utc':datetime.now(timezone.utc).isoformat(),'setup':str(setup),'run':str(run),
        'target_rounds':10000,'save_every':100,'validation_every':100,'optimization':'history priority with25% uniform exploration; tail guard conditional on benchmark'}
if a.exit_code is not None:record['exit_code']=a.exit_code
for name in ('group_totals.json','progress.json'):
 if (run/name).exists():record[name.removesuffix('.json')]=json.loads((run/name).read_text())
for path in (setup/'status.json',setup.parent/'dapo_active_run.json'):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
print(json.dumps(record),flush=True)
