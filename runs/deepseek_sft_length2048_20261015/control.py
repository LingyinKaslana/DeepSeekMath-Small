import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--phase',required=True)
parser.add_argument('--status',required=True)
parser.add_argument('--exit-code',type=int)
args=parser.parse_args()
setup=Path(__file__).resolve().parent
record={'status':args.status,'phase':args.phase,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'setup':str(setup),'max_new_tokens':2048,'batch_size':128,
        'selection':'Each model own original length-limited predictions only',
        'total_rerun_questions':561,'counts':{'sft':561},
        'sft_progress':str(setup/'sft/progress.json'),
        'report':str(setup/'report/comparison.json'),'training_modified':False}
if args.exit_code is not None:record['exit_code']=args.exit_code
if args.status=='complete_verified':
    result=json.loads((setup/'report/comparison.json').read_text())
    assert result['status']=='complete_verified'
    record['comparison']=result['comparison']
for path in (setup/'status.json',setup.parent/'deepseek_math_inference_active_run.json'):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
print(json.dumps(record,ensure_ascii=False),flush=True)
