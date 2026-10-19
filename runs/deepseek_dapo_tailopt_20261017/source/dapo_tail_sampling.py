"""Biased difficulty routing with explicit exploration and causal EOS tail guards."""
import math,random
from collections import defaultdict
import torch
from transformers.cache_utils import DynamicCache
from group_length_shadow import GroupLengthShadow


class DifficultyRouter:
    def __init__(self,history=None,exploration=.25,pool_multiplier=8,strategy='weighted'):
        assert 0<exploration<=1 and pool_multiplier>=1
        self.history=history if history is not None else {}
        self.exploration=exploration;self.pool_multiplier=pool_multiplier
        assert strategy in ('weighted','rank');self.strategy=strategy

    def choose(self,pool,number,seed):
        assert len(pool)>=number>0
        rng=random.Random(seed);pool=list(pool);rng.shuffle(pool)
        uniform=max(1,math.ceil(number*self.exploration));chosen=pool[:uniform]
        candidates=pool[uniform:]
        # Weighted sampling, not a fixed top-k: no question has zero priority.
        def priority(row):
            item=self.history.get(row['question_sha256'],{});p=item.get('p',.5)
            weight=.02+4*p*(1-p)
            return math.log(max(rng.random(),1e-30))/weight if self.strategy=='weighted' else weight
        candidates.sort(key=priority,reverse=True)
        return chosen+candidates[:number-uniform]

    def observe(self,group,round_index):
        key=group[0]['question_sha256'];old=self.history.get(key,{})
        p=sum(row.get('raw_correct',row['correct']) for row in group)/len(group)
        self.history[key]={'p':.7*old.get('p',.5)+.3*p,
                           'visits':old.get('visits',0)+1,'last_round':round_index}


@torch.inference_mode()
def tail_generate(model,input_ids,attention_mask,eos_token_id,max_new_tokens,
                  samples,finish_correct,relative=True,batch_tail=False,
                  minimum_tokens=512,tail_multiplier=2.):
    """Compact native decode; cancellation is explicit and never invents EOS.

    EOS events are graded first; relative references are actual completed EOS
    responses. Batch threshold uses only already finished lengths. Cancelled
    responses are returned as incomplete negative examples by the caller.
    """
    device=input_ids.device;count,width=input_ids.shape
    assert count%samples==0
    output=torch.full((count,max_new_tokens),eos_token_id,dtype=torch.long,device=device)
    original=torch.arange(count,device=device);ids=input_ids
    lengths=[None]*count;reasons=[None]*count;details={};finished_lengths=[]
    guards=[GroupLengthShadow(size=samples,min_correct=2,multiplier=2.,minimum_tokens=256)
            for _ in range(count//samples)]
    kwargs={'past_key_values':DynamicCache(),'attention_mask':attention_mask,
            'cache_position':torch.arange(width,device=device),'use_cache':True}
    for step in range(max_new_tokens):
        prepared=model.prepare_inputs_for_generation(ids,**kwargs)
        result=model(**prepared,return_dict=True,logits_to_keep=1)
        logits=result.logits[:,-1,:].to(copy=True,dtype=torch.float32)
        token=torch.multinomial(logits.softmax(-1),num_samples=1).squeeze(1)
        output[original,step]=token
        kwargs=model._update_model_kwargs_for_generation(result,kwargs,is_encoder_decoder=False)
        ids=torch.cat((ids,token[:,None]),dim=-1)
        events=defaultdict(list)
        ended_positions=torch.nonzero(token==eos_token_id,as_tuple=True)[0]
        ended_ids=original.index_select(0,ended_positions).cpu().tolist()
        for index in ended_ids:
            lengths[index]=step+1;reasons[index]='eos';finished_lengths.append(step+1)
            if relative:
                correct=finish_correct(index,output[index,:step+1].cpu().tolist())
                events[index//samples].append({'index':index%samples,'length':step+1,
                                               'correct':correct,'complete_eos':True})
        cancelled=[]
        if relative:
            for group_index,guard in enumerate(guards):
                for event in guard.observe(step+1,events.get(group_index,())):
                    index=group_index*samples+event['sample_index']
                    if lengths[index] is None:
                        lengths[index]=step+1;reasons[index]='group_relative_length';cancelled.append(index)
                        details[index]={**event,'shadow_only':False,'forced_incorrect':True}
        if batch_tail and len(finished_lengths)>=math.ceil(.75*count):
            ordered=sorted(finished_lengths)
            quantile=ordered[math.ceil(.9*len(ordered))-1]
            threshold=max(minimum_tokens,tail_multiplier*quantile)
            if step+1>threshold:
                for index in original.cpu().tolist():
                    if lengths[index] is None:
                        lengths[index]=step+1;reasons[index]='batch_tail';cancelled.append(index)
                        details[index]={'threshold':threshold,'finished_p90':quantile,'forced_incorrect':True}
        keep_mask=token!=eos_token_id
        if cancelled:
            keep_mask&=~torch.isin(original,torch.tensor(cancelled,device=device,dtype=torch.long))
        keep=torch.nonzero(keep_mask,as_tuple=True)[0]
        if keep.numel()==0:break
        if keep.numel()!=ids.shape[0]:
            ids=ids.index_select(0,keep);original=original.index_select(0,keep)
            kwargs['attention_mask']=kwargs['attention_mask'].index_select(0,keep)
            kwargs['past_key_values'].batch_select_indices(keep)
        del result,logits
    for index in range(count):
        if lengths[index] is None:lengths[index]=step+1;reasons[index]='length'
    return {'tokens':output[:,:step+1],'lengths':lengths,'finish_reasons':reasons,'cancellations':details}
