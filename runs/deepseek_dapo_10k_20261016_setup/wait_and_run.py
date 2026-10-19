import json,time,os,subprocess
from pathlib import Path
setup=Path(__file__).resolve().parent;plan=json.loads((setup/'plan.json').read_text())
status=Path(plan['wait_for'])
while True:
    prior=json.loads(status.read_text())
    if prior['status']=='complete_verified':break
    if prior['status']=='failed':
        raise RuntimeError('Prior benchmark failed; inspect before sharing GPU: '+str(prior))
    subprocess.run([os.sys.executable,str(setup/'control.py'),'--phase','waiting_for_100r_benchmark','--status','queued'],check=True)
    time.sleep(30)
# The completed pipeline no longer has generation work pending; independently check GPU ownership.
while True:
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if not gpu:break
    time.sleep(5)
(setup/'launch.json').write_text(json.dumps({'started_unix':time.time(),'resume_round':100,'tmux_session':'deepseek-dapo-10k-20261016'},indent=2)+'\n')
os.execv('/bin/bash',['bash',str(setup/'run.sh')])
