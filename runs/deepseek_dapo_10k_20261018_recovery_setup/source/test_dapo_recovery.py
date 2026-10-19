import unittest
from unittest.mock import patch
import torch
from dapo_math import capture_round_state,restore_round_state
class RecoveryTests(unittest.TestCase):
 def test_mutating_rollout_cannot_change_saved_history_cursor_or_rng(self):
  state={'completed_rounds':2200,'training_order_cursor':100,'difficulty_history':{'question':{'p':.5}},'effective_groups_used':70400}
  torch.manual_seed(123)
  with patch('torch.cuda.get_rng_state',return_value=torch.tensor([42],dtype=torch.uint8)),patch('torch.cuda.set_rng_state') as setter:
   snapshot=capture_round_state(state);expected=torch.rand(3);torch.rand(10)
   state['training_order_cursor']+=256;state['difficulty_history']['question']['p']=1.
   restored=restore_round_state(snapshot)
   self.assertEqual(restored['training_order_cursor'],100);self.assertEqual(restored['difficulty_history']['question']['p'],.5)
   self.assertEqual(restored['completed_rounds'],2200);self.assertTrue(torch.equal(torch.rand(3),expected));setter.assert_called_once()
if __name__=='__main__':unittest.main()
