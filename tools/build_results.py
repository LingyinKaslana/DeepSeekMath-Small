"""Build read-only summary tables and standalone plots from retained numerical logs."""
import csv,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def lines(name):
 with (ROOT/name).open() as stream:return [json.loads(line) for line in stream]
def build():
 docs=ROOT/'docs';figures=docs/'figures';figures.mkdir(parents=True,exist_ok=True)
 cpt=read('runs/deepseek_coder_1p3b_10b_20261012/result.json');sft=read('runs/deepseek_sft_gsm_math_20261015/result.json')
 rl=lines('runs/deepseek_dapo_10k_20261016/training.jsonl');val=read('runs/deepseek_dapo_10k_20261016/validation_history.json');progress=read('runs/deepseek_dapo_10k_20261016/progress.json')
 evaluations=[]
 for label,path in [('Original DeepSeek-Coder-1.3B','runs/deepseek_coder_base_inference_20261015/scores.json'),('CPT10B','runs/deepseek_coder_10b_inference_20261015/scores.json'),('SFT best step904','runs/deepseek_sft_gsm_math_20261015_setup/evaluation/scores.json'),('Early RL round80','runs/deepseek_dapo_100r_20261015_setup_v5/evaluation/scores.json')]:
  scores=read(path)['benchmarks'];evaluations.append({'checkpoint':label,'gsm8k':scores['gsm8k']['correct'],'math':scores['math']['correct'],'source':path})
 recent=rl[-100:];seconds=(datetime.fromisoformat(recent[-1]['utc'])-datetime.fromisoformat(recent[0]['utc'])).total_seconds()/(len(recent)-1)
 snapshot=read('EXPORT_MANIFEST.json');best=max(val,key=lambda x:x['accuracy']);probe=read('runs/deepseek_dapo_10k_20261017_tail_setup/benchmark.json')
 result={'snapshot_started_utc':snapshot['snapshot_started_utc'],'snapshot_finished_utc':snapshot['snapshot_finished_utc'],
 'cpt':{'status':cpt['status'],'input_tokens':cpt['trained_tokens'],'initial_validation_loss':cpt['validation_before']['loss'],'final_validation_loss':cpt['validation_after']['loss'],'steps':cpt['completed_steps']},
 'sft':{k:sft[k] for k in ('status','completed_steps','sft_input_tokens','sft_supervised_tokens','best_step','best_validation_loss')},
 'rl':{'status':'in_progress_at_snapshot','completed_rounds':rl[-1]['round'],'last_completed_utc':rl[-1]['utc'],'target_rounds':10000,
       'gradient_updates':sum(r['updates'] for r in rl),'effective_groups_found':sum(r['counts']['mixed'] for r in rl),'effective_groups_used':len(rl)*32,
       'best_validation_round':best['round'],'best_validation_accuracy':best['accuracy'],'latest_validation':val[-1],
       'last100_seconds_per_round':seconds,'last100_rounds_per_hour':3600/seconds,'last100_mixed_fraction':sum(r['counts']['mixed'] for r in recent)/sum(r['candidate_groups'] for r in recent),
       'snapshot_progress':progress},'full_test_1024':evaluations,
 'sampling_probe_256':probe['summary'],'discarded_formal_branches':read('runs/dapo_execution_ledger.json')}
 (docs/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 text=f'''# 结果快照\n\n快照窗口（UTC）：{result['snapshot_started_utc']} ～ {result['snapshot_finished_utc']}。训练工作区仍在运行；本目录是静态记录。\n\n## 主线结果\n\n| 阶段 | 状态与数值 |\n|---|---|\n| DeepSeek CPT | {cpt['trained_tokens']:,} input token；验证 loss {cpt['validation_before']['loss']:.4f} → {cpt['validation_after']['loss']:.4f} |\n| SFT | {sft['completed_steps']:,} steps；最佳 step {sft['best_step']}，验证 loss {sft['best_validation_loss']:.4f} |\n| RL | 已记录 {rl[-1]['round']:,} / 10,000 轮；{result['rl']['gradient_updates']:,} 次梯度更新；实际使用 {result['rl']['effective_groups_used']:,} 个有效组 |\n\n当前 RL 轨迹计数不包含回退后丢弃的正式训练分支。相关执行量见 `runs/dapo_execution_ledger.json`；短跑的 disposable 更新单独标记。\n\n## 完整测试，统一 1,024 token\n\n| 检查点 | GSM8K 正确数 / 1,319 | MATH 正确数 / 5,000 |\n|---|---:|---:|\n'''
 for item in evaluations:text+=f"| {item['checkpoint']} | {item['gsm8k']} ({item['gsm8k']/1319:.2%}) | {item['math']} ({item['math']/5000:.2%}) |\n"
 text+='\n这些是原始完整测试记录。选择性 2,048 token 补测另见 `runs/deepseek_length2048_20261015/report/` 和 `runs/deepseek_sft_length2048_20261015/report/`，不混入上表。\n\n## 长 RL 的独立验证曲线\n\n'
 text+=f"256 题 held-out GSM8K，zero-shot / greedy / EOS-only / 2,048 token。最佳第 {best['round']} 轮：{best['correct']}/256 = {best['accuracy']:.2%}。最近第 {val[-1]['round']} 轮：{val[-1]['correct']}/256 = {val[-1]['accuracy']:.2%}。\n\n"
 text+='| 轮数 | 正确数 / 256 | 正确率 |\n|---|---:|---:|\n'
 for point in val[-8:]:text+=f"| {point['round']} | {point['correct']} | {point['accuracy']:.2%} |\n"
 text+=f"\n最近 100 轮平均 {seconds:.1f} 秒/轮，约 {3600/seconds:.1f} 轮/小时；候选中混合对错组占 {result['rl']['last100_mixed_fraction']:.2%}。这只描述快照前的实际速度，不推断后续恒定吞吐。\n\n"
 text+='![RL validation](figures/rl_validation.png)\n\n## 短期性能对照\n\n同一第 1,900 轮权重，正式 256 路采样，每次凑齐 32 个有效组：\n\n| 策略 | 耗时 | 有效组比例 |\n|---|---:|---:|\n'
 for trial in probe['trials']:text+=f"| {trial['policy']} | {trial['seconds']:.2f} s | {trial['mixed_fraction']:.2%} |\n"
 text+='\n历史优先路由的这一短期有效组产出速度约为 1.88 倍。训练继续后大量题目变为全对，有效组比例下降，长期速度并未保持这一倍率。64 题、4 update 的 execution probe 也不能证明能力保持。\n\n## CPT / SFT 验证\n\n![CPT validation](figures/cpt_validation.png)\n\n![SFT validation](figures/sft_validation.png)\n\n## 阅读数值时的限制\n\n单种子、小模型、缩小语料；GSM8K 训练题已用于 SFT 与 RL；只有所述精确重叠检查；CPT 没有完整测试集去污染。当前最佳 RL 验证检查点尚不能视为完整测试成绩。早期 RL 检查点的 MATH 测试分数下降也保留在上表。\n'
 (docs/'results.md').write_text(text)
 with (docs/'experiments.csv').open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['experiment','family','status_at_snapshot','completed_rounds_or_steps','files','bytes'])
  for directory in sorted((ROOT/'runs').iterdir()):
   if not directory.is_dir():continue
   name=directory.name;family='RL' if ('dapo' in name or 'group_length' in name) else 'SFT' if 'sft' in name else 'DeepSeek CPT/eval' if 'deepseek' in name else 'Qwen CPT/eval/diagnostic'
   status={}
   for candidate in ('result.json','status.json','progress.json','manifest.json'):
    if (directory/candidate).exists():
     value=json.loads((directory/candidate).read_text())
     if isinstance(value,dict):status=value;break
   files=[p for p in directory.rglob('*') if p.is_file()]
   writer.writerow([name,family,status.get('status','recorded'),status.get('completed_rounds',status.get('completed_steps','')),len(files),sum(p.stat().st_size for p in files)])
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
 except ImportError as exc:raise SystemExit('Install requirements-analysis.txt to build figures') from exc
 plt.rcParams.update({'figure.dpi':150,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','svg'):fig.savefig(figures/f'{name}.{ext}')
  plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));ax.plot([p['round'] for p in val],[100*p['accuracy'] for p in val],'-o',markersize=3)
 ax.scatter([best['round']],[100*best['accuracy']],color='tab:red',zorder=4,label=f"Best: round {best['round']}, {100*best['accuracy']:.2f}%")
 ax.axvline(1900,color='tab:orange',linestyle='--',alpha=.6,label='History-priority branch starts')
 ax.set(xlabel='Completed rollout rounds',ylabel='Held-out GSM8K accuracy (%)',title='1.3B DAPO: 256-question zero-shot validation');ax.legend(fontsize=8);save(fig,'rl_validation')
 cval=read('runs/deepseek_coder_1p3b_10b_20261012/validation_history.json');cval=[{'step':0,**cpt['validation_before']}]+cval+[cpt['validation_after']]
 fig,ax=plt.subplots(figsize=(8,4));ax.plot([p['step']/1000 for p in cval],[p['loss'] for p in cval]);ax.set(xlabel='Optimizer steps (thousands)',ylabel='Validation next-token loss',title='DeepSeek-Coder-1.3B: mathematical CPT, 10B input tokens');save(fig,'cpt_validation')
 sval=read('runs/deepseek_sft_gsm_math_20261015/validation_history.json');fig,ax=plt.subplots(figsize=(8,4));ax.plot([p['step'] for p in sval],[p['loss'] for p in sval],'-o',markersize=3)
 ax.axvline(sft['best_step'],color='tab:red',linestyle='--',label=f"Selected step {sft['best_step']}");ax.set(xlabel='Optimizer steps',ylabel='Answer-only validation loss',title='GSM8K + MATH SFT: 512-question validation');ax.legend();save(fig,'sft_validation')
 print(json.dumps({'status':'complete','rl_completed_rounds':rl[-1]['round'],'full_test_models':len(evaluations)}))
if __name__=='__main__':build()
