"""Replay recorded eight-answer groups using only causally available EOS labels."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from group_length_shadow import replay_group


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize(c):
    d=dict(c)
    for output,numerator,denominator in (
        ('flagged_response_percent','flagged_responses','responses'),
        ('flagged_final_correct_percent','flagged_final_correct','flagged_responses'),
        ('all_correct_responses_flagged_percent','flagged_final_correct','correct_responses'),
        ('potential_generated_token_saving_percent','potential_saved_tokens','generated_tokens'),
        ('flagged_group_percent','flagged_groups','groups'),
        ('length_capped_responses_caught_percent','flagged_length_capped','length_capped_responses')):
        d[output]=100*c[numerator]/c[denominator] if c[denominator] else None
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.run/'rollouts.jsonl';digest=sha(source)
    result=json.loads((a.run/'result.json').read_text());assert result['status']=='complete_verified' and result['completed_rounds']==100
    aggregate=Counter();windows=defaultdict(Counter);flag_lengths=[]
    settings={'min_correct':2,'multiplier':2.,'minimum_tokens':256}
    with source.open() as reader,(a.output/'flagged_responses.jsonl').open('w') as writer:
        for index,line in enumerate(reader):
            group=json.loads(line);samples=group['samples'];assert len(samples)==8
            flags=replay_group(samples,**settings)
            counts=Counter(groups=1,responses=8,generated_tokens=sum(len(r['tokens']) for r in samples),
                correct_responses=sum(r['correct'] for r in samples),
                length_capped_responses=sum(r['finish_reason']=='length' for r in samples),
                flagged_groups=bool(flags),flagged_responses=len(flags),
                flagged_final_correct=sum(f['actual_final_correct'] for f in flags),
                flagged_final_incorrect=sum(not f['actual_final_correct'] for f in flags),
                flagged_length_capped=sum(f['actual_finish_reason']=='length' for f in flags),
                potential_saved_tokens=sum(f['potential_saved_tokens'] for f in flags),
                accepted_groups=bool(group['accepted']),
                accepted_flagged_groups=bool(group['accepted'] and flags),
                all_correct_groups_becoming_mixed_if_flags_marked_wrong=bool(flags and all(r['correct'] for r in samples)))
            aggregate.update(counts);windows[(group['round']-1)//25].update(counts)
            for flag in flags:
                sample=samples[flag['sample_index']]
                record={'group_record_index':index,'round':group['round'],'group_accepted_for_training':group['accepted'],
                    **flag,'actual_completion':sample['completion'],
                    'reference_completions':[samples[i]['completion'] for i in flag['reference_samples']],
                    'all_group_lengths':[len(r['tokens']) for r in samples],
                    'all_group_correctness':[r['correct'] for r in samples]}
                writer.write(json.dumps(record,ensure_ascii=False)+'\n');flag_lengths.append(flag)
    assert aggregate['responses']==result['sampled_responses']
    assert aggregate['generated_tokens']==result['sampled_response_tokens']
    assert sha(source)==digest
    summary={'status':'complete_verified','completed_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'Offline causal shadow replay; no actual cancellation, reward changes or parameter updates',
        'source_run':str(a.run.resolve()),'source_rollouts_sha256':digest,
        'source_result_sha256':sha(a.run/'result.json'),'settings':settings,
        'event_order':'Process EOS and cap completions first; then flag still-active samples only when generated_length > max(256,2*median(completed_correct_EOS_lengths))',
        'virtual_cancellation':'Flagged samples retain real generated output for evaluation but cannot become future reference answers',
        'aggregate':summarize(aggregate),
        'by_round_window':{f'{25*k+1}-{25*k+25}':summarize(v) for k,v in sorted(windows.items())},
        'top_potential_savings':sorted(flag_lengths,key=lambda r:r['potential_saved_tokens'],reverse=True)[:10],
        'limits':['Frozen recorded trajectories; actual cancellation changes RNG consumption, batch rounding and potentially future outputs.',
            'Potential token saving is not an observed wall-clock speedup.',
            'Final correctness is revealed to the rule only at EOS; full final correctness is used only to audit flags.',
            'Long-but-correct flags measure a tradeoff, not proof the trajectory was unhelpful.',
            'Uses train questions already seen in SFT/RL; does not estimate held-out generalization.'],
        'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ('audit_group_length_shadow.py','group_length_shadow.py')}}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'aggregate':summary['aggregate'],'by_round_window':summary['by_round_window']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
