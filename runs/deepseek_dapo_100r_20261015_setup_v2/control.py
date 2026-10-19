import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from dapo_math import write
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--status',required=True)
p.add_argument('--exit-code',type=int);a=p.parse_args()
setup=Path(__file__).resolve().parent
record={'status':a.status,'phase':a.phase,'utc':datetime.now(timezone.utc).isoformat(),
        'setup':str(setup),'run':str(setup.parent/'deepseek_dapo_100r_20261015'),
        'rounds':100,'effective_groups_per_round':32,'responses_per_group':8,'accepted_response_budget':25600}
if a.exit_code is not None:record['exit_code']=a.exit_code
write(setup/'status.json',record);write(setup.parent/'dapo_active_run.json',record)
print(json.dumps(record),flush=True)
