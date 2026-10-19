import json,os,signal,time,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd();run=root/'runs/deepseek_dapo_100r_20261015';out=root/'runs/deepseek_dapo_tuning_20261015'
pid=1850589;parent=1849165
assert 'deepseek_dapo_100r_20261015_setup_v2/source/dapo_math.py' in Path(f'/proc/{pid}/cmdline').read_text()
while True:
    checkpoint=run/'latest_checkpoint'
    try:metadata=json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
    if metadata['completed_rounds']>=55:
        os.kill(pid,signal.SIGSTOP);os.kill(parent,signal.SIGSTOP)
        # The checkpoint directory has already been atomically committed.
        snapshot=out/'resume_checkpoint'
        shutil.copytree(checkpoint,snapshot,copy_function=os.link)
        with (run/'training.jsonl').open() as f:rows=[json.loads(l) for l in f]
        assert rows[-1]['round']==metadata['completed_rounds'], 'Unexpected uncheckpointed completed round'
        record={'status':'captured','utc':datetime.now(timezone.utc).isoformat(),'completed_rounds':metadata['completed_rounds'],
          'checkpoint':str(snapshot),'checkpoint_manifest_sha256':hashlib.sha256((snapshot/'checkpoint_manifest.json').read_bytes()).hexdigest(),
          'old_pid':pid,'old_launcher_pid':parent,'last_completed_training_round':rows[-1]['round'],
          'optimizer_and_rng_preserved':True,'no_completed_round_rollback':True,'original_config':metadata['config']}
        (out/'boundary.json').write_text(json.dumps(record,indent=2)+'\n')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);os.kill(parent,signal.SIGKILL)
        print(json.dumps(record),flush=True);break
    time.sleep(1)
