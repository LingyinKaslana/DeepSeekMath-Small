"""CPU-only boundary and audit checks for the LR overlay helper."""
from copy import deepcopy
import math
import unittest

from lr_intervention import intervention_lr, validate_intervention, validate_intervention_audit


def policy_fixture():
    return {"schema_version": 1, "type": "linear_input_tokens", "start_input_tokens": 5_000_000_000,
            "end_input_tokens": 6_000_000_000, "start_lr": 3e-6, "end_lr": 3e-7, "after": "hold"}


def state_fixture():
    schedule = {"total_steps": 305176, "peak_lr": 3e-6, "warmup_steps": 15259,
                "decay_80_step": 244141, "decay_90_step": 274659,
                "decay_80_factor": .316, "decay_90_factor": .1}
    policy = policy_fixture()
    return {"completed_steps": 140_000, "trained_tokens": 4_587_520_000,
            "schedule": schedule, "lr_intervention": policy,
            "lr_intervention_audit": {"policy": deepcopy(policy), "applied_at_completed_steps": 120_000,
                                      "applied_at_input_tokens": 3_932_160_000,
                                      "resumed_checkpoint": "/fixture/checkpoint_step_00120000",
                                      "saved_utc": "2026-10-10T00:00:00+00:00",
                                      "original_schedule": deepcopy(schedule), "reason": "User requested a 5B to 6B linear LR reduction",
                                      "extra_metadata": {"authorized": True}}}


class InterventionTests(unittest.TestCase):
    def test_exact_start_end_and_holding(self):
        policy = policy_fixture()
        for tokens in (0, 32_768, 5_000_000_000 - 32_768, 5_000_000_000):
            with self.subTest(tokens=tokens):
                self.assertEqual(intervention_lr(policy, tokens), 3e-6)
        for tokens in (6_000_000_000, 6_000_000_000 + 32_768, 10_000_000_000):
            with self.subTest(tokens=tokens):
                self.assertEqual(intervention_lr(policy, tokens), 3e-7)

    def test_linear_interior_and_adjacent_update_boundaries(self):
        policy = policy_fixture()
        for tokens in (5_000_000_000 + 32_768, 5_250_000_000, 5_500_000_000,
                       5_750_000_000, 6_000_000_000 - 32_768):
            with self.subTest(tokens=tokens):
                expected = 3e-6 - (tokens - 5_000_000_000) * 2.7e-6 / 1_000_000_000
                self.assertTrue(math.isclose(intervention_lr(policy, tokens), expected, rel_tol=1e-14, abs_tol=1e-20))
        self.assertLess(intervention_lr(policy, 5_000_000_000 + 32_768), 3e-6)
        self.assertGreater(intervention_lr(policy, 6_000_000_000 - 32_768), 3e-7)

    def test_pre_update_count_is_used(self):
        policy = policy_fixture()
        before = 5_000_000_000 - 16_384
        self.assertEqual(intervention_lr(policy, before), 3e-6)
        self.assertLess(intervention_lr(policy, before + 32_768), 3e-6)

    def test_no_policy_is_explicit(self):
        self.assertIsNone(validate_intervention(None))
        self.assertIsNone(validate_intervention_audit({}))
        self.assertIsNone(validate_intervention_audit({"lr_intervention": None, "lr_intervention_audit": None}))
        with self.assertRaises(ValueError):
            intervention_lr(None, 0)
        for audit in ({}, [], {"policy": None}):
            with self.subTest(audit=audit), self.assertRaises(ValueError):
                validate_intervention_audit({"lr_intervention_audit": audit})

    def test_policy_strict_schema(self):
        policy = policy_fixture()
        for key in policy:
            invalid = dict(policy)
            invalid.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_intervention(invalid)
        for extra in ("start_tokens", "reason", "unexpected"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_intervention(dict(policy, **{extra: 1}))
        for key, value in (("schema_version", True), ("schema_version", 1.0), ("schema_version", 2),
                           ("type", "linear_steps"), ("after", "decay"),
                           ("start_input_tokens", -1), ("start_input_tokens", 0),
                           ("end_input_tokens", -1), ("end_input_tokens", 5_000_000_000),
                           ("start_input_tokens", 6_000_000_000), ("start_input_tokens", True),
                           ("start_input_tokens", 5e9), ("end_input_tokens", float("inf"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_intervention(dict(policy, **{key: value}))

    def test_nonfinite_and_nonpositive_lr_rejected(self):
        for key in ("start_lr", "end_lr"):
            for value in (float("nan"), float("inf"), float("-inf"), -3e-7, 0, True, "3e-6", 10 ** 1000):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_intervention(dict(policy_fixture(), **{key: value}))

    def test_input_count_rejected(self):
        for value in (-1, True, 1.0, float("nan"), float("inf"), "5000000000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                intervention_lr(policy_fixture(), value)

    def test_valid_audit_and_nonmutation(self):
        state = state_fixture()
        before = deepcopy(state)
        validated = validate_intervention_audit(state)
        self.assertEqual(validated, before["lr_intervention_audit"])
        validated["original_schedule"]["peak_lr"] = 1e-6
        validated["policy"]["end_lr"] = 1e-6
        self.assertEqual(state, before)
        self.assertEqual(validate_intervention(policy_fixture()), policy_fixture())

    def test_audit_must_match_policy_and_original_schedule(self):
        state = state_fixture()
        bad = deepcopy(state)
        bad["lr_intervention_audit"]["policy"]["end_lr"] = 1e-7
        with self.assertRaises(ValueError):
            validate_intervention_audit(bad)
        bad = deepcopy(state)
        bad["lr_intervention_audit"]["original_schedule"]["peak_lr"] = 1e-6
        with self.assertRaises(ValueError):
            validate_intervention_audit(bad)
        bad = deepcopy(state)
        bad["lr_intervention_audit"]["original_schedule"]["peak_lr"] = float("nan")
        with self.assertRaises(ValueError):
            validate_intervention_audit(bad)

    def test_audit_application_bounds_and_history(self):
        for key, value in (("applied_at_completed_steps", -1), ("applied_at_completed_steps", True),
                           ("applied_at_completed_steps", 140_001), ("applied_at_input_tokens", -1),
                           ("applied_at_input_tokens", 4_587_520_001)):
            state = state_fixture()
            state["lr_intervention_audit"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_intervention_audit(state)
        state = state_fixture()
        state["trained_tokens"] = 6_000_000_000
        state["lr_intervention_audit"]["applied_at_input_tokens"] = 5_000_000_001
        with self.assertRaises(ValueError):
            validate_intervention_audit(state)
        state["lr_intervention_audit"]["applied_at_input_tokens"] = 5_000_000_000
        self.assertIsInstance(validate_intervention_audit(state), dict)

    def test_missing_or_malformed_audit_rejected(self):
        state = state_fixture()
        for key in state["lr_intervention_audit"]:
            if key == "extra_metadata":
                continue
            bad = deepcopy(state)
            bad["lr_intervention_audit"].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_intervention_audit(bad)
        for audit in (None, [], "audit"):
            bad = deepcopy(state)
            bad["lr_intervention_audit"] = audit
            with self.subTest(audit=audit), self.assertRaises(ValueError):
                validate_intervention_audit(bad)
        for key, value in (("reason", " "), ("resumed_checkpoint", ""), ("saved_utc", "not a date"),
                           ("saved_utc", "2026-10-10T00:00:00"),
                           ("saved_utc", "2026-10-10T08:00:00+08:00")):
            bad = deepcopy(state)
            bad["lr_intervention_audit"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_intervention_audit(bad)
        state["lr_intervention_audit"]["saved_utc"] = "2026-10-10T00:00:00Z"
        self.assertIsInstance(validate_intervention_audit(state), dict)


if __name__ == "__main__":
    unittest.main()
