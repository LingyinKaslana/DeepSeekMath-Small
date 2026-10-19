"""Auditable token-based LR overlay; the original training schedule stays saved.

Call ``intervention_lr`` with cumulative input tokens *before* the next update.
An absent policy leaves schedule selection to the trainer.  This module reads
no checkpoints, imports no training framework, and does not mutate its inputs.
"""
from copy import deepcopy
from datetime import datetime, timedelta
import math


POLICY_FIELDS = frozenset({"schema_version", "type", "start_input_tokens", "end_input_tokens",
                           "start_lr", "end_lr", "after"})
AUDIT_FIELDS = frozenset({"policy", "applied_at_completed_steps", "applied_at_input_tokens",
                          "resumed_checkpoint", "saved_utc", "original_schedule", "reason"})


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_json(value, name):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must not contain nonfinite numbers")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} must have string keys")
        for key, item in value.items():
            _finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{name}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"{name} must contain only JSON-compatible values")


def validate_intervention(policy):
    """Return a copied schema-v1 policy, or None; reject extra/missing keys.

    Token boundaries are positive integer counts.  LR values are finite,
    strictly positive JSON numbers.  ``start_input_tokens < end_input_tokens``
    is required; only linear interpolation followed by holding is supported.
    """
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError("LR intervention policy must be a dictionary or None")
    missing = POLICY_FIELDS - policy.keys()
    extra = policy.keys() - POLICY_FIELDS
    if missing or extra:
        raise ValueError(f"LR intervention policy keys differ from schema: missing={sorted(missing)}, extra={sorted(extra)}")
    if isinstance(policy["schema_version"], bool) or not isinstance(policy["schema_version"], int) or policy["schema_version"] != 1:
        raise ValueError("LR intervention schema_version must be integer 1")
    if policy["type"] != "linear_input_tokens":
        raise ValueError("LR intervention type must be linear_input_tokens")
    if policy["after"] != "hold":
        raise ValueError("LR intervention after must be hold")
    start = _integer(policy["start_input_tokens"], "start_input_tokens", minimum=1)
    end = _integer(policy["end_input_tokens"], "end_input_tokens", minimum=1)
    if start >= end:
        raise ValueError("start_input_tokens must be smaller than end_input_tokens")
    start_lr = _positive_finite(policy["start_lr"], "start_lr")
    end_lr = _positive_finite(policy["end_lr"], "end_lr")
    return dict(policy, start_lr=start_lr, end_lr=end_lr)


def intervention_lr(policy, input_tokens):
    """Compute the next update's LR from its pre-update input-token count."""
    policy = validate_intervention(policy)
    if policy is None:
        raise ValueError("A policy is required; the caller retains the original schedule when no policy is present")
    tokens = _integer(input_tokens, "input_tokens")
    start, end = policy["start_input_tokens"], policy["end_input_tokens"]
    if tokens <= start:
        return policy["start_lr"]
    if tokens >= end:
        return policy["end_lr"]
    fraction = (tokens - start) / (end - start)
    return policy["start_lr"] * (1.0 - fraction) + policy["end_lr"] * fraction


def validate_intervention_audit(state):
    """Validate ``state.lr_intervention`` and its audit; return a copied audit.

    Extra audit metadata is allowed.  The overlay's application point must be
    within the saved completed prefix and no later than the token boundary at
    which LR starts changing.  The original schedule must stay unchanged.
    Absence of a policy requires absence (or None) of its audit.
    """
    if not isinstance(state, dict):
        raise ValueError("Training state must be a dictionary")
    policy = validate_intervention(state.get("lr_intervention"))
    audit = state.get("lr_intervention_audit")
    if policy is None:
        if audit is not None:
            raise ValueError("LR intervention audit cannot exist without a policy")
        return None
    if not isinstance(audit, dict):
        raise ValueError("An LR intervention policy requires a dictionary audit")
    missing = AUDIT_FIELDS - audit.keys()
    if missing:
        raise ValueError(f"LR intervention audit is missing {sorted(missing)}")
    audited_policy = validate_intervention(audit["policy"])
    if audited_policy != policy:
        raise ValueError("LR intervention audit policy differs from the saved policy")
    completed = _integer(state.get("completed_steps"), "completed_steps")
    trained = _integer(state.get("trained_tokens"), "trained_tokens")
    applied_step = _integer(audit["applied_at_completed_steps"], "applied_at_completed_steps")
    applied_tokens = _integer(audit["applied_at_input_tokens"], "applied_at_input_tokens")
    if applied_step > completed or applied_tokens > trained:
        raise ValueError("LR intervention application point exceeds the saved completed prefix")
    if applied_tokens > policy["start_input_tokens"]:
        raise ValueError("LR intervention must be applied by start_input_tokens to avoid rewriting LR history")
    _nonempty_string(audit["resumed_checkpoint"], "resumed_checkpoint")
    _nonempty_string(audit["reason"], "reason")
    saved_utc = _nonempty_string(audit["saved_utc"], "saved_utc")
    try:
        timestamp = datetime.fromisoformat(saved_utc.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("saved_utc must be an ISO timestamp in UTC") from None
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("saved_utc must include the UTC offset or Z")
    original_schedule = audit["original_schedule"]
    if not isinstance(original_schedule, dict) or not original_schedule:
        raise ValueError("original_schedule must be a nonempty dictionary")
    _finite_json(original_schedule, "original_schedule")
    if original_schedule != state.get("schedule"):
        raise ValueError("LR intervention audit original_schedule differs from the saved training schedule")
    result = deepcopy(audit)
    result["policy"] = audited_policy
    return result
