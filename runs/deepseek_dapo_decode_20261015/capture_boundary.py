import json,os,signal,time,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd();run=root/'runs/deepseek_dapo_100r_20261015';out=root/'runs/deepseek_dapo_decode_20261015'
pid=1881317;parent=1881213
while True:
    checkpoint=run/'latest_checkpoint'
    try:m=json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
    if m['completed_rounds']>=75:
        os.kill(pid,signal.SIGSTOP);os.kill(parent,signal.SIGSTOP)
        try:
            with (run/'training.jsonl').open() as f:rows=[json.loads(l) for l in f]
            if rows[-1]['round']!=m['completed_rounds']:
                os.kill(parent,signal.SIGCONT);os.kill(pid,signal.SIGCONT);time.sleep(1);continue
            shutil.copytree(checkpoint,out/'resume_checkpoint',copy_function=os.link)
            record={'status':'captured','utc':datetime.now(timezone.utc).isoformat(),'completed_rounds':m['completed_rounds'],
                'checkpoint':str(out/'resume_checkpoint'),'old_pid':pid,'old_launcher_pid':parent,'no_completed_round_rollback':True}
            (out/'boundary.json').write_text(json.dumps(record,indent=2)+'\n')
        except BaseException:
            os.kill(parent,signal.SIGCONT);os.kill(pid,signal.SIGCONT);raise
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);os.kill(parent,signal.SIGKILL)
        print(json.dumps(record),flush=True);break
    time.sleep(1)
