import argparse,json
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--status',required=True);p.add_argument('--exit-code',type=int);a=p.parse_args()
setup=Path(__file__).resolve().parent;plan=json.loads((setup/'plan.json').read_text());run=Path(plan['run'])
record={'status':a.status,'phase':a.phase,'utc':datetime.now(timezone.utc).isoformat(),
        'setup':str(setup),'run':str(run),'target_rounds':10000,'save_every':100,'validation_every':100,
        'effective_groups_per_round':32,'responses_per_group':8,'target_used_groups':320000,'target_updates':40000}
if a.exit_code is not None:record['exit_code']=a.exit_code
if (run/'group_totals.json').exists():record['group_totals']=json.loads((run/'group_totals.json').read_text())
else:record['group_totals']={'completed_rounds':100,**plan['starting_group_totals']}
for path in (setup/'status.json',setup.parent/'dapo_active_run.json'):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
print(json.dumps(record),flush=True)
