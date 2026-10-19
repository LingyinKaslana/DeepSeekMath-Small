import unittest
from unittest.mock import patch
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from dapo_tail_sampling import DifficultyRouter,tail_generate
from compact_decode import compact_generate

class TailSamplingTests(unittest.TestCase):
 def test_priority_is_reproducible_has_exploration_and_tracks_raw_results(self):
  rows=[{'question_sha256':str(i)} for i in range(1000)]
  history={str(i):{'p':.5 if i<100 else .9999} for i in range(1000)}
  router=DifficultyRouter(history,exploration=.25)
  result=router.choose(rows,100,seed=42)
  self.assertEqual(result,router.choose(rows,100,seed=42))
  self.assertEqual(len({r['question_sha256'] for r in result}),100)
  self.assertGreater(sum(int(r['question_sha256'])<100 for r in result),40)
  self.assertTrue(any(int(r['question_sha256'])>=100 for r in result))
  router.observe([{'question_sha256':'0','correct':False,'raw_correct':True}]*8,20)
  self.assertAlmostEqual(history['0']['p'],.65)

 def tiny(self):
  torch.manual_seed(12);torch.set_num_threads(1)
  return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024,eos_token_id=2,pad_token_id=2)).eval()

 def test_disabled_guard_matches_compacted_sampler_and_rng(self):
  model=self.tiny();ids=torch.tensor([[3,4],[5,6],[7,8],[9,10]]);mask=torch.ones_like(ids)
  torch.manual_seed(42);a=compact_generate(model,ids,mask,2,20)
  torch.manual_seed(42);b=tail_generate(model,ids,mask,2,20,2,lambda *_:False,relative=False)
  for i,length in enumerate(b['lengths']):self.assertTrue(torch.equal(a[i,:length],b['tokens'][i,:length]))

 def test_relative_guard_cancels_after_completed_correct_references(self):
  model=self.tiny();ids=torch.tensor([[3,4]]*8);mask=torch.ones_like(ids);calls=0
  def sample(probabilities,num_samples):
   nonlocal calls
   calls+=1;out=torch.full((probabilities.shape[0],1),3,dtype=torch.long)
   if calls==1:out[:2]=2
   return out
  with patch('torch.multinomial',side_effect=sample):
   result=tail_generate(model,ids,mask,2,300,8,lambda *_:True,relative=True)
  self.assertEqual(result['finish_reasons'][:2],['eos']*2)
  self.assertEqual(result['lengths'][2:],[257]*6)
  self.assertEqual(result['finish_reasons'][2:],['group_relative_length']*6)
  for index in range(2,8):self.assertNotIn(2,result['tokens'][index,:result['lengths'][index]].tolist())

 def test_batch_tail_uses_only_observed_eos_and_respects_floor(self):
  model=self.tiny();ids=torch.tensor([[3,4]]*8);mask=torch.ones_like(ids);calls=0
  def sample(probabilities,num_samples):
   nonlocal calls
   calls+=1;out=torch.full((probabilities.shape[0],1),3,dtype=torch.long)
   if calls==2:out[:6]=2
   return out
  with patch('torch.multinomial',side_effect=sample):
   result=tail_generate(model,ids,mask,2,30,8,lambda *_:False,relative=False,batch_tail=True,minimum_tokens=10)
  self.assertEqual(result['lengths'][-2:],[11,11]);self.assertEqual(result['finish_reasons'][-2:],['batch_tail']*2)

if __name__=='__main__':unittest.main()
