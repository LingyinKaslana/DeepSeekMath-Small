"""Numerical checks for reward boundaries, PPO clipping and token weighting."""
import unittest
import torch
from dapo_math import numeric_answer, grade, shaped_reward, advantages, surrogate, save_checkpoint
from pathlib import Path
import json
import tempfile
from unittest.mock import patch


class DapoTests(unittest.TestCase):
    def test_checkpoint_commit_with_stale_previous_directory(self):
        class FakeModel:
            def save_pretrained(self,path,**kwargs):(path/'weight.txt').write_text('saved')
        class FakeTokenizer:
            def save_pretrained(self,path):(path/'tokenizer.json').write_text('{}')
        class FakeOptimizer:
            def state_dict(self):return {'state':{},'param_groups':[]}
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.get_rng_state',return_value=torch.tensor([42],dtype=torch.uint8)):
            path=Path(directory)/'latest_checkpoint'
            stale=Path(directory)/'latest_checkpoint.previous';stale.mkdir();(stale/'proof').write_text('keep')
            for step in (1,2):
                save_checkpoint(path,FakeModel(),FakeTokenizer(),FakeOptimizer(),{'completed_rounds':step})
                self.assertEqual(json.loads((path/'checkpoint_manifest.json').read_text())['completed_rounds'],step)
                payload=torch.load(path/'training_state.pt',weights_only=True)
                self.assertEqual(payload['completed_rounds'],step)
            self.assertEqual((stale/'proof').read_text(),'keep')
            self.assertFalse(path.with_name(path.name+'.incomplete').exists())
            self.assertEqual(list(Path(directory).glob('*.previous.*')),[])

    def test_numeric_reward_no_guess_overwrite(self):
        self.assertTrue(grade('work\nThe answer is 1,200.', '1200'))
        self.assertTrue(grade('The answer is -0.50.', '-.5'))
        self.assertFalse(grade('The answer is 3.\nThe answer is 2.', '2'))
        self.assertFalse(grade('The answer is 2 or 3.', '2'))
        self.assertFalse(grade('There are 2 possible answers.', '2'))
        self.assertIsNone(numeric_answer('The answer is NaN.'))

    def test_soft_length_boundaries_and_empty_signal(self):
        self.assertEqual(shaped_reward(True,1536),1.)
        self.assertEqual(shaped_reward(True,1792),.5)
        self.assertEqual(shaped_reward(False,2048),-2.)
        self.assertEqual(shaped_reward(False,20),-1.)
        self.assertTrue(torch.equal(advantages([-1.]*8),torch.zeros(8)))
        values=advantages([-1.,1.,1.,-1.])
        self.assertAlmostEqual(float(values.mean()),0.)
        self.assertAlmostEqual(float(values.std(correction=0)),1.,places=5)

    def test_asymmetric_clip_gradient(self):
        new=torch.tensor([.1, .5, -.1, -.5],dtype=torch.float64,requires_grad=True)
        old=torch.zeros(4,dtype=torch.float64)
        adv=torch.tensor([1.,1.,-1.,-1.],dtype=torch.float64)
        loss,_=surrogate(new,old,adv)
        loss.sum().backward()
        self.assertAlmostEqual(float(new.grad[0]),-float(torch.exp(new.detach()[0])))
        self.assertEqual(float(new.grad[1]),0.)
        self.assertAlmostEqual(float(new.grad[2]),float(torch.exp(new.detach()[2])))
        self.assertEqual(float(new.grad[3]),0.)
        # Independent finite difference away from clipping boundaries.
        for i in range(4):
            plus=new.detach().clone();minus=plus.clone();plus[i]+=1e-5;minus[i]-=1e-5
            finite=(surrogate(plus,old,adv)[0].sum()-surrogate(minus,old,adv)[0].sum())/2e-5
            self.assertAlmostEqual(float(finite),float(new.grad[i]),places=7)

    def test_token_weighting_micro_accumulation(self):
        # Unequal lengths: per-response mean would incorrectly weight these equally.
        parameter=torch.tensor(.05,dtype=torch.float64,requires_grad=True)
        tokens=[torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([-1.]*6,dtype=torch.float64)]
        for values in tokens:
            loss,_=surrogate(parameter*values,torch.zeros_like(values),values.sign())
            (loss.sum()/8).backward()
        accumulated=parameter.grad.clone();parameter.grad=None
        values=torch.cat(tokens)
        loss,_=surrogate(parameter*values,torch.zeros_like(values),values.sign())
        loss.mean().backward()
        self.assertTrue(torch.allclose(accumulated,parameter.grad,atol=1e-12,rtol=0))


if __name__=='__main__':unittest.main()
