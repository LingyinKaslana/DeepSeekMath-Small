import os,time,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
setup=Path(__file__).resolve().parent;boundary=setup.parent/'deepseek_dapo_decode_20261015/boundary.json'
while not boundary.exists():time.sleep(1)
now=datetime.now(timezone.utc)
(setup/'launch.json').write_text(json.dumps({'started_utc':now.isoformat(),'started_beijing':now.astimezone(timezone(timedelta(hours=8))).isoformat(),'tmux_session':'deepseek-dapo-100r-v5-20261015','resume_round':json.loads(boundary.read_text())['completed_rounds'],'pipeline':'Native-vs-compacted decoding benchmark + numerical checks + disposable4update smoke -> restore original checkpoint and finish100 rounds -> full benchmarks'},indent=2)+'\n')
os.execv('/bin/bash',['bash',str(setup/'run.sh')])
