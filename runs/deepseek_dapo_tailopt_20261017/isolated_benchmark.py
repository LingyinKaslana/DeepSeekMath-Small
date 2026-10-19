import os,signal,subprocess,json,time
from pathlib import Path
setup=Path(__file__).resolve().parent;root=setup.parent.parent;pid=1936427
assert 'deepseek_dapo_10k_20261016_setup/source/dapo_math.py' in Path(f'/proc/{pid}/cmdline').read_text()
record={'main_pid':pid,'paused_unix':time.time(),'completed_rounds':json.loads((root/'runs/deepseek_dapo_10k_20261016/progress.json').read_text())['completed_rounds']}
os.kill(pid,signal.SIGSTOP)
try:
 time.sleep(3)
 assert '\nState:\tT' in Path(f'/proc/{pid}/status').read_text()
 (setup/'main_pause.json').write_text(json.dumps(record,indent=2)+'\n')
 with (setup/'benchmark.log').open('w') as stream:
  completed=subprocess.run([str(root/'.venv/bin/python'),'-u',str(setup/'source/benchmark_dapo_tailopt.py'),'--setup',str(setup)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
 record['benchmark_exit_code']=completed.returncode
finally:
 os.kill(pid,signal.SIGCONT);record['resumed_unix']=time.time()
 (setup/'main_pause.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
