"""Numerical checks for reward boundaries, PPO clipping and token weighting."""
import unittest
import torch
from dapo_math import numeric_answer, grade, shaped_reward, advantages, surrogate


class DapoTests(unittest.TestCase):
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
