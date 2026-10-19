"""CPU checks for explicit CPT schedules, strict resume, and native Llama gating."""
import argparse
import math
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cpt_orders import prefix_counts
from train_cpt import check_model_safety, learning_rate, make_schedule, validate_resume
from verify_cpt_checkpoint import expected_schedule, last_lr


class FakeDataset:
    sequence_length = 4
    tokens = list(range(41))

    def __len__(self):
        return math.ceil(len(self.tokens) / self.sequence_length)


class ScheduleTests(unittest.TestCase):
    def test_historical_default_schema_and_boundaries_are_unchanged(self):
        expected = {"total_steps": 100, "peak_lr": 3e-6, "warmup_steps": 5,
                    "decay_80_step": 80, "decay_90_step": 90,
                    "decay_80_factor": .316, "decay_90_factor": .1}
        self.assertEqual(make_schedule(100, 3e-6), expected)
        self.assertEqual(expected_schedule(100, 3e-6), expected)
        self.assertAlmostEqual(learning_rate(expected, 0), 3e-6 / 5, places=20)
        self.assertEqual(learning_rate(expected, 79), 3e-6)
        self.assertEqual(learning_rate(expected, 80), 3e-6 * .316)
        self.assertEqual(learning_rate(expected, 90), 3e-6 * .1)

    def test_constant_schedule_holds_through_both_historical_decay_boundaries(self):
        schedule = make_schedule(100, 3e-6, 7, "constant")
        self.assertEqual(schedule, expected_schedule(100, 3e-6, 7, "constant"))
        for step in (0, 6, 7, 79, 80, 89, 90, 99):
            rate = 3e-6 * min((step + 1) / 7, 1)
            self.assertEqual(learning_rate(schedule, step), rate)
            self.assertEqual(last_lr(schedule, step + 1), rate)
        self.assertEqual(make_schedule(15259, 3e-6, lr_schedule="constant")["warmup_steps"], 763)

    def test_zero_warmup_and_invalid_policies(self):
        schedule = make_schedule(10, 3e-6, 0, "constant")
        self.assertTrue(all(learning_rate(schedule, k) == 3e-6 for k in range(10)))
        for warmup in (-1, 11, True, 1.5):
            with self.assertRaises(ValueError):
                make_schedule(10, 3e-6, warmup, "constant")
            with self.assertRaises(ValueError):
                expected_schedule(10, 3e-6, warmup, "constant")
        with self.assertRaises(ValueError):
            make_schedule(10, 3e-6, lr_schedule="cosine")

    def test_resume_rejects_schedule_and_warmup_changes(self):
        dataset = FakeDataset()
        order = list(range(len(dataset)))
        signature = {"effective_batch": 3, "epochs": 1,
                     "lr_schedule": "constant", "warmup_steps": 2}
        args = argparse.Namespace(epochs=1, steps=None, lr=3e-6, seed=42,
                                  lr_schedule="constant", warmup_steps=2)
        inputs, targets, updates = prefix_counts(order, 2, 3, 4, 41)
        state = {"schema_version": 1, "signature": signature, "order": order,
                 "schedule": make_schedule(4, args.lr, 2, "constant"), "completed_steps": 2,
                 "trained_tokens": inputs, "supervised_tokens": targets, "optimizer_updates": updates}
        validate_resume(state, signature, dataset, args)
        for change in ({"lr_schedule": "multistep"}, {"warmup_steps": 1}, {"warmup_steps": None}):
            changed_args = argparse.Namespace(**(vars(args) | change))
            with self.assertRaisesRegex(ValueError, "schedule"):
                validate_resume(state, signature, dataset, changed_args)
            changed_signature = signature | change
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                validate_resume(state, changed_signature, dataset, args)
        tampered = dict(state, schedule=dict(state["schedule"], decay_90_factor=.1))
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_resume(tampered, signature, dataset, args)


class NativeLlamaTests(unittest.TestCase):
    def model(self):
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        config._attn_implementation = "sdpa"
        return LlamaForCausalLM(config)

    def test_native_llama_is_allowed_but_configured_path_remains_closed(self):
        model = self.model()
        check_model_safety(model, native_standard=True)
        with self.assertRaisesRegex(ValueError, "standard_native mode only"):
            check_model_safety(model)

    def test_noncausal_configuration_and_wrong_parameter_precision_are_rejected(self):
        model = self.model()
        model.config.sliding_window = 128
        with self.assertRaisesRegex(ValueError, "full causal SDPA"):
            check_model_safety(model, native_standard=True)
        model.config.sliding_window = None
        model.to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "FP32 master"):
            check_model_safety(model, native_standard=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
