"""Native HF decode with EOS-completed sequences removed from the active batch.

Same model, cache dtypes, position preparation, softmax/multinomial and EOS
semantics as the unwarped HF sampler. Compaction changes RNG consumption and
batch-shape rounding; it does not change temperature, reward or output cap.
"""
import torch
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def compact_generate(model,input_ids,attention_mask,eos_token_id,max_new_tokens,
                     do_sample=True,compact=True):
    device=input_ids.device;count,width=input_ids.shape
    output=torch.full((count,max_new_tokens),eos_token_id,dtype=torch.long,device=device)
    original=torch.arange(count,device=device)
    unfinished=torch.ones(count,dtype=torch.bool,device=device)
    kwargs={'past_key_values':DynamicCache(),'attention_mask':attention_mask,
            'cache_position':torch.arange(width,device=device),'use_cache':True}
    ids=input_ids
    for step in range(max_new_tokens):
        prepared=model.prepare_inputs_for_generation(ids,**kwargs)
        result=model(**prepared,return_dict=True,logits_to_keep=1)
        logits=result.logits[:,-1,:].to(copy=True,dtype=torch.float32)
        if do_sample:next_token=torch.multinomial(logits.softmax(-1),num_samples=1).squeeze(1)
        else:next_token=logits.argmax(-1)
        if not compact:next_token=torch.where(unfinished,next_token,eos_token_id)
        output[original,step]=next_token
        kwargs=model._update_model_kwargs_for_generation(result,kwargs,is_encoder_decoder=False)
        ids=torch.cat((ids,next_token[:,None]),dim=-1)
        if compact:
            keep=torch.nonzero(next_token!=eos_token_id,as_tuple=True)[0]
            if keep.numel()==0:break
            if keep.numel()!=ids.shape[0]:
                ids=ids.index_select(0,keep);original=original.index_select(0,keep)
                kwargs['attention_mask']=kwargs['attention_mask'].index_select(0,keep)
                kwargs['past_key_values'].batch_select_indices(keep)
        else:
            unfinished=unfinished&(next_token!=eos_token_id)
            if not unfinished.any():break
        del result,logits
    return output[:,:step+1]
