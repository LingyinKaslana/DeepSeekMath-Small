"""Causal shadow guard for relative length, with no generation cancellation.

Only completed EOS answers can become correctness references. A hypothetical
cancelled answer is excluded from later references, while its actual full
trajectory remains available for measuring missed correct answers.
"""
import math
from statistics import median


class GroupLengthShadow:
    def __init__(self,size=8,min_correct=2,multiplier=2.,minimum_tokens=256):
        assert size>0 and 1<=min_correct<=size and multiplier>0 and minimum_tokens>=0
        self.size=size;self.min_correct=min_correct;self.multiplier=multiplier
        self.minimum_tokens=minimum_tokens;self.finished=set();self.references={}
        self.would_stop={};self.last_step=0

    def threshold(self):
        if len(self.references)<self.min_correct:return None
        return max(self.minimum_tokens,self.multiplier*median(self.references.values()))

    def observe(self,step,finished=()):
        assert step>self.last_step
        self.last_step=step
        # Process this step's EOS before testing unfinished responses.
        for event in finished:
            index=event['index'];assert 0<=index<self.size and index not in self.finished
            assert event['length']==step
            self.finished.add(index)
            if index not in self.would_stop and event['correct'] and event['complete_eos']:
                self.references[index]=step
        threshold=self.threshold();new=[]
        if threshold is not None and step>threshold:
            for index in range(self.size):
                if index in self.finished or index in self.would_stop:continue
                record={'sample_index':index,'would_stop_at_token':step,'threshold':threshold,
                    'reference_median_tokens':median(self.references.values()),
                    'reference_samples':sorted(self.references),
                    'reference_lengths':[self.references[i] for i in sorted(self.references)],
                    'reason':'group_relative_length','shadow_only':True}
                self.would_stop[index]=record;new.append(record)
        return new


def replay_group(samples,**settings):
    guard=GroupLengthShadow(size=len(samples),**settings)
    events={}
    for index,sample in enumerate(samples):
        length=len(sample['tokens']);assert length>0
        events.setdefault(length,[]).append({'index':index,'length':length,
            'correct':sample['correct'],'complete_eos':sample['finish_reason']=='eos'})
    # Jump only to completion events or the earliest possible threshold crossing.
    # No future correctness is passed to the guard before its EOS step.
    while events:
        step=min(events)
        threshold=guard.threshold()
        active=len(guard.finished|set(guard.would_stop))<guard.size
        if threshold is not None and active:
            step=min(step,max(guard.last_step+1,math.floor(threshold)+1))
        guard.observe(step,events.pop(step,()))
    result=[]
    for index,flag in sorted(guard.would_stop.items()):
        sample=samples[index];length=len(sample['tokens'])
        result.append({**flag,'actual_tokens':length,'potential_saved_tokens':length-flag['would_stop_at_token'],
            'actual_final_correct':sample['correct'],'actual_finish_reason':sample['finish_reason'],
            'source_id':sample.get('source_id'),'expected':sample.get('expected'),
            'actual_extracted':sample.get('extracted')})
    return result
