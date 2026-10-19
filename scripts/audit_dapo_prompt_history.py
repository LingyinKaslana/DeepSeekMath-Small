"""Forward-only chronological audit of historical difficulty-based selection."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--through-round',type=int);a=p.parse_args()
history={};buckets=defaultdict(lambda:dict(groups=0,mixed=0,all_correct=0,all_wrong=0,tokens=0))
with (a.run/'rollouts.jsonl').open() as stream:
 for line in stream:
  try:record=json.loads(line)
  except json.JSONDecodeError:break # Live final physical line may be in flight.
  if a.through_round and record['round']>a.through_round:break
  group=record['samples'];key=group[0]['question_sha256'];n=sum(x['correct'] for x in group);size=len(group)
  old=history.get(key);prob=old['p'] if old else .5
  score=4*prob*(1-prob)
  if record['round']>=1500:
   bucket='high' if score>=.64 else 'medium' if score>=.25 else 'low'
   row=buckets[bucket];row['groups']+=1;row['mixed']+=0<n<size;row['all_correct']+=n==size;row['all_wrong']+=n==0;row['tokens']+=sum(len(x['tokens']) for x in group)
  history[key]={'p':.7*prob+.3*n/size,'visits':(old['visits'] if old else 0)+1,'last_round':record['round']}
for row in buckets.values():row['mixed_fraction']=row['mixed']/row['groups'];row['tokens_per_mixed']=row['tokens']/max(row['mixed'],1)
result={'status':'complete','history_questions':len(history),'buckets':dict(buckets),'history':history,'scope':'Each record scored using preceding groups only; not a causal estimate of future training benefit.'}
a.output.mkdir(parents=True,exist_ok=True);(a.output/'history_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history'}))
