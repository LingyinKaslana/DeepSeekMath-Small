"""CPU-only small-tensor audit; does not load a model or train."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/cpt_native_from0_5h_20261009"
OLD = json.loads((ROOT / "runs/cpt_validation_curve_20261009/weight_decay_diagnostics.json").read_text())
BASE = Path(OLD["base_snapshot"])
NAMES = [row["name"] for row in OLD["sampled_weights"]["results"]]
RESULT = json.loads((RUN / "result.json").read_text())
SCHEDULE = RESULT["schedule"]


def read_small(directory):
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {name: "model.safetensors" for name in NAMES}
    result = {}
    for shard in sorted({weight_map[name] for name in NAMES}):
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in NAMES:
                if weight_map[name] == shard:
                    tensor = handle.get_tensor(name)
                    assert tensor.numel() == 1536
                    result[name] = tensor.float().numpy().copy()
    return result


def lr(step):
    # Zero-based update exactly mirrors the frozen learning_rate() schedule.
    factor = (step + 1) / SCHEDULE["warmup_steps"] if step < SCHEDULE["warmup_steps"] else 1.0
    assert step < SCHEDULE["decay_80_step"]
    return float(np.float32(SCHEDULE["peak_lr"] * factor))


base = read_small(BASE)
flat_base = np.concatenate([base[name] for name in NAMES]).astype(np.float32)
simulated = flat_base.copy()
sum_logs = 0.0
sum_lr = 0.0
check_steps = [5000, 15259, 20000, 25000, 30000, RESULT["completed_steps"]]
pure = []
sim_snapshots = {}
coefficient_difference_count = 0
for step in range(RESULT["completed_steps"]):
    rate = lr(step)
    # Native foreach CPU scalar is cast to FP32 when multiplying FP32 parameters.
    native_coef = np.float32(1.0 - rate * 0.1)
    tensor_coef = np.float32(np.float32(1.0) - np.float32(np.float32(rate) * np.float32(0.1)))
    coefficient_difference_count += int(native_coef != tensor_coef)
    np.multiply(simulated, native_coef, out=simulated)
    sum_lr += rate
    sum_logs += math.log1p(-0.1 * rate)
    if step + 1 in check_steps:
        multiplier = math.exp(sum_logs)
        pure.append({"steps": step + 1, "sum_applied_lr": sum_lr,
                     "real_arithmetic_multiplier": multiplier,
                     "real_arithmetic_shrink_percent": (1.0 - multiplier) * 100.0,
                     "fp32_small_tensor_combined_l2_ratio": float(np.linalg.norm(simulated.astype(np.float64)) / np.linalg.norm(flat_base.astype(np.float64))),
                     "fp32_relative_deviation_from_real_counterfactual": float(np.linalg.norm(simulated.astype(np.float64) - flat_base.astype(np.float64) * multiplier) / np.linalg.norm(flat_base.astype(np.float64)))})
        sim_snapshots[step + 1] = simulated.copy()

checkpoints = []
for steps, name in [(20000, "checkpoint_step_00020000"), (25000, "checkpoint_step_00025000"),
                    (30000, "checkpoint_step_00030000"), (RESULT["completed_steps"], "stopped_checkpoint")]:
    directory = RUN / name
    saved = read_small(directory)
    real_factor = next(row["real_arithmetic_multiplier"] for row in pure if row["steps"] == steps)
    rows = []
    offset = 0
    for parameter in NAMES:
        before = base[parameter].astype(np.float64)
        after = saved[parameter].astype(np.float64)
        simulated_after = sim_snapshots[steps][offset:offset + before.size].astype(np.float64)
        offset += before.size
        denominator = np.linalg.norm(before)
        rows.append({"name": parameter, "elements": int(before.size),
                     "base_mean": float(before.mean()), "checkpoint_mean": float(after.mean()),
                     "checkpoint_l2_to_base": float(np.linalg.norm(after) / denominator),
                     "relative_l2_change": float(np.linalg.norm(after - before) / denominator),
                     "relative_l2_residual_vs_fp32_decay_only": float(np.linalg.norm(after - simulated_after) / denominator),
                     "fp32_decay_only_l2_to_base": float(np.linalg.norm(simulated_after) / denominator)})
    checkpoints.append({"steps": steps, "checkpoint": str(directory), "real_decay_only_multiplier": real_factor, "small_tensors": rows})

state = torch.load(RUN / "stopped_checkpoint/training_state.pt", map_location="cpu", mmap=True, weights_only=True)
optimizer_key = next(k for k in ["optimizer", "optimizer_state", "optimizer_state_dict"] if k in state)
optimizer = state[optimizer_key]
groups = [{key: value for key, value in group.items() if key != "params"} | {"parameter_count": len(group["params"])}
          for group in optimizer["param_groups"]]
assert len(groups) == 1 and groups[0]["parameter_count"] == 338 and groups[0]["weight_decay"] == .1

report = {
    "schema_version": 1, "status": "completed_cpu_small_tensor_review",
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_execution": False, "training_configuration_modified": False,
    "run": str(RUN), "base_snapshot": str(BASE),
    "loading": "safetensors.safe_open(cpu) reads eight 1536-element tensors from each saved model; optimizer file torch.load(cpu,mmap=True,weights_only=True), inspect group metadata only",
    "optimizer_groups": groups,
    "schedule": SCHEDULE, "effective_batch_tokens": 32768,
    "parameter_groups": {"total_unique_tensors": 338, "rmsnorm_weights": 57, "attention_biases": 84, "other_weight_tensors_including_shared_embedding": 197},
    "decay_only_counterfactual": {"gradient_derived_adam_updates": "omitted", "applied_lr": "LR rounded to FP32 exactly as set_native_lr before each update", "real_formula": "exp(sum(log1p(-weight_decay * applied_lr)))", "fp32_simulation": "CPU NumPy FP32 multiplication of the eight original small tensors, once per update; no gradients, not a proof of exact CUDA bitwise behavior", "results": pure},
    "native_scalar_vs_tensor_lr_decay_coefficient": {"examined_updates": RESULT["completed_steps"], "different_fp32_coefficients": coefficient_difference_count, "interpretation": "Compares FP32(1 - double(FP32(lr))*.1) to FP32(1 - FP32(FP32(lr)*FP32(.1))). This covers the coefficient only, not all Adam update precision."},
    "checkpoint_small_tensors": checkpoints,
    "installed_hf_trainer_default": {"version": "4.55.4", "source": str(ROOT / ".venv/lib/python3.12/site-packages/transformers/trainer.py") + ":1189", "behavior": "Trainer filters bias, LayerNorm, RMSNorm/name norm out of the decay group; assigns those parameters weight_decay=0. Current custom trainer passes model.parameters() as one WD=.1 group.", "paper_policy": "Paper does not disclose bias/norm decay grouping; this is not evidence our grouping differs from the paper."},
    "interpretation": ["The native 1.011B curve confirms early loss rebound without optimized execution. Norm/bias decay is a recipe hypothesis, not a diagnosed implementation bug.", "Pure decay at the 5000-step loss minimum is only about .082%; it cannot alone establish an explanation for the entire later loss increase.", "Actual ratios include gradients. Similarity to a pure-decay ratio is association, and nontrivial residuals show gradient effects remain.", "The eight tensors are diagnostic samples, not a census or causal measurement of their effect on loss.", "FP32 parameter multiplication slightly differs from real arithmetic; a CPU counterfactual does not attribute training loss to that rounding."],
    "minimal_causal_intervention": {"preferred": "Matched native training arms from identical base or complete checkpoint. Preserve weights, Adam states, next data batches, RNG, LR schedule and clipping. One arm keeps all-parameter WD=.1; the other sets bias/RMSNorm WD=0 while weight matrices retain WD=.1. Evaluate same fixed train probe and validation at matched updates/tokens.", "window": "5000-10000 matched updates for a directional result; from base the strongest early loss minimum comparator requires reaching ~5000-10000 steps.", "avoid": "Do not change LR and decay grouping in the same arm. Do not copy one-period optimizer moments into a differently initialized run or reset schedule. Counterfactual weight rescaling at inference is not equivalent to training without decay.", "secondary": "Separate lower-LR arm if grouping test fails or as a first priority because loss rebounds while LR increases; retain decay unchanged."}
}
out = Path(__file__).with_name("decay_review.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(out), "decay": pure, "latest_tensor_ratios": [{"name": row["name"], "ratio": row["checkpoint_l2_to_base"], "decay_residual": row["relative_l2_residual_vs_fp32_decay_only"]} for row in checkpoints[-1]["small_tensors"]], "coeff_differences": coefficient_difference_count}, ensure_ascii=False))
