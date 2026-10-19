import unittest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,GenerationConfig
from compact_decode import compact_generate


def trim(rows,eos=2):
    result=[]
    for row in rows.tolist():
        result.append(row[:row.index(eos)+1] if eos in row else row)
    return result


class CompactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        config=LlamaConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=4,vocab_size=16,eos_token_id=2,pad_token_id=2)
        config._attn_implementation='sdpa'
        self.model=LlamaForCausalLM(config).eval()
        self.ids=torch.tensor([[2,2,1,4],[2,1,3,5],[1,4,6,7],[2,2,1,8]])
        self.mask=torch.tensor([[0,0,1,1],[0,1,1,1],[1,1,1,1],[0,0,1,1]])

    def reference(self,sample):
        config=GenerationConfig(max_new_tokens=32,do_sample=sample,
            top_k=0 if sample else None,top_p=1. if sample else None,
            eos_token_id=2,pad_token_id=2,use_cache=True)
        return self.model.generate(input_ids=self.ids,attention_mask=self.mask,generation_config=config,logits_to_keep=1)[:,4:]

    def test_same_greedy_and_cache_positions(self):
        with torch.inference_mode():
            expected=trim(self.reference(False))
            actual=trim(compact_generate(self.model,self.ids,self.mask,2,32,do_sample=False))
        self.assertEqual(expected,actual)

    def test_sampler_matches_hf_without_compaction(self):
        torch.manual_seed(123);expected=trim(self.reference(True))
        torch.manual_seed(123);actual=trim(compact_generate(self.model,self.ids,self.mask,2,32,compact=False))
        self.assertEqual(expected,actual)
        self.assertGreater(len({len(row) for row in expected}),1)

    def test_compacted_sampling_eos_and_order(self):
        torch.manual_seed(123);actual=trim(compact_generate(self.model,self.ids,self.mask,2,32))
        self.assertEqual(len(actual),4)
        self.assertTrue(all(1<=len(row)<=32 for row in actual))
        self.assertTrue(all(2 not in row[:-1] for row in actual))
        self.assertGreater(len({len(row) for row in actual}),1)
        # No row has exited before first generated token: identical initial draws.
        torch.manual_seed(123);reference=trim(self.reference(True))
        self.assertEqual([r[0] for r in reference],[r[0] for r in actual])

    def test_bfloat16_autocast_sampler_matches_native(self):
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            torch.manual_seed(456);expected=trim(self.reference(True))
            torch.manual_seed(456);actual=trim(compact_generate(self.model,self.ids,self.mask,2,32,compact=False))
        self.assertEqual(expected,actual)


if __name__=='__main__':unittest.main()
