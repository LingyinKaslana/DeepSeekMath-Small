"""Read-only CPU checks of the actual production orders and resumed LR policy."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile

SETUP = Path(__file__).resolve().parents[1]
PROJECT = SETUP.parents[1]
sys.path.insert(0, str(SETUP / "source"))
import numpy as np
import torch
import train_cpt as train
from cpt_orders import load_saved_orders, prefix_counts
from lr_intervention import validate_intervention, validate_intervention_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint or Path(json.loads((PROJECT / "runs/cpt_10b_lr3e6_20261009/latest_checkpoint.json").read_text())["path"])
    state = torch.load(checkpoint / "training_state.pt", mmap=True, map_location="cpu", weights_only=True)
    orders = load_saved_orders(state, checkpoint)
    config = train.load_backend(SETUP / "backend_config.json")
    policy = validate_intervention(json.loads((SETUP / "lr_intervention.json").read_text()))
    assert state["trained_tokens"] < policy["start_input_tokens"]
    assert train.cpt.sha(SETUP / "backend_config.json") == state["signature"]["backend_config_sha256"]
    hashes = {name: train.cpt.sha(SETUP / "source" / name) for name in train.SOURCE_FILES}
    signature = dict(state["signature"], source_sha256=hashes)
    tokens = np.load(PROJECT / "data/processed/openwebmath_5b_s42/train_tokens.npy", mmap_mode="r", allow_pickle=False)

    class Dataset:
        sequence_length = config["sequence_length"]
        def __init__(self, tokens):
            self.tokens = tokens
        def __len__(self):
            return math.ceil(len(self.tokens) / self.sequence_length)

    dataset = Dataset(tokens)
    arguments = argparse.Namespace(allow_source_migration=True, allow_native_backend_migration=False,
                                   resume=checkpoint, backend_config=SETUP / "backend_config.json",
                                   epochs=2, seed=42, steps=None, lr=3e-6)
    train.validate_resume(state, signature, dataset, arguments, orders)
    total = prefix_counts(orders, state["schedule"]["total_steps"], 16, 2048, len(tokens))[0]
    assert total == 10_000_000_000
    candidate = {k: copy.deepcopy(v) for k, v in state.items() if k not in {"optimizer", "cpu_rng", "cuda_rng"}}
    with tempfile.TemporaryDirectory() as tmp:
        train.configure_lr_intervention(candidate, SETUP / "lr_intervention.json", checkpoint, Path(tmp), total)
        validate_intervention_audit(candidate)
        train.validate_resume(candidate, signature, dataset, arguments, orders)
        before = copy.deepcopy(candidate["lr_intervention_audit"])
        train.configure_lr_intervention(candidate, SETUP / "lr_intervention.json", checkpoint, Path(tmp), total)
        assert candidate["lr_intervention_audit"] == before
        train.configure_lr_intervention(candidate, None, checkpoint, Path(tmp), total)
        assert candidate["lr_intervention_audit"] == before
    rates = []
    for completed in [state["completed_steps"], 152587, 152588, 152589, 167847, 179208, 179209,
                      183105, 183106, 183107, 244141, 274659, 305175]:
        consumed = prefix_counts(orders, completed, 16, 2048, len(tokens))[0]
        rate = train.effective_learning_rate(candidate, completed, consumed)
        fraction = min(1., max(0., (consumed - 5_000_000_000) / 1_000_000_000))
        expected = 3e-6 * (1. - fraction) + 3e-7 * fraction
        assert math.isclose(rate, expected, rel_tol=1e-14, abs_tol=1e-20)
        rates.append({"next_update": completed + 1, "pre_update_input_tokens": consumed,
                      "mathematical_lr": rate, "expected_fp32_lr": float(np.float32(rate))})
    assert prefix_counts(orders, 152588, 16, 2048, len(tokens))[0] == 5_000_000_000
    assert prefix_counts(orders, 183106, 16, 2048, len(tokens))[0] == 6_000_012_288
    assert candidate["schedule"] == state["schedule"]
    unchanged = [name for name in train.SOURCE_FILES if name in state["signature"]["source_sha256"]
                 and hashes[name] == state["signature"]["source_sha256"][name]]
    assert set(unchanged) == set(state["signature"]["source_sha256"]) - {"train_cpt.py"}
    result = {"status": "passed", "cpu_only": True, "optimizer_updates_performed": 0,
              "checked_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
              "resumed_steps": state["completed_steps"], "resumed_input_tokens": state["trained_tokens"],
              "configured_input_tokens": total, "policy": policy, "original_schedule_preserved": True,
              "old_order_metadata_preserved": candidate["order_metadata"] == state["order_metadata"],
              "source_sha256": hashes, "unchanged_original_sources": unchanged,
              "backend_sha256": signature["backend_config_sha256"], "rates": rates,
              "scope": "Actual order-prefix token accounting, original resume guards, policy inheritance and future-only application; no model load or GPU update."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checkpoint": str(checkpoint),
                      "resumed_input_tokens": result["resumed_input_tokens"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
