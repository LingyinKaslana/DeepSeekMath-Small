"""Train or resume CPT epochs using a separately validated local backend.

Forward uses BF16 autocast; master parameters, gradients and AdamW moments stay
FP32. Stops preserve the original schedule. Explicit resumed LR interventions
are saved separately and use the cumulative input-token count before an update.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import signal
import time
import traceback

import torch
from transformers import AutoTokenizer

import cpt
from cpt_orders import EpochOrders, load_saved_orders, order_indices, prefix_counts, validate_permutation
from compiled_adamw import make_compiled_adamw, set_lr
from model_backends import configure_model, eager_native
from nonfinite_retry import checked_optimizer_update
from packed_data import PackedTokens
from source_migration import (apply_source_migration, apply_native_migration, authorize_signature,
                              authorize_native_signature, native_configs, validate_lineage, verify_lineage_snapshots)
from standard_native import (check_native_config, load_native_model, make_native_optimizer,
                             adapt_native_optimizer, set_native_lr, update_native, prepare_train_probe, preserve_rng)
from token_batches import PackedTokenBatches
from lr_intervention import validate_intervention, intervention_lr, validate_intervention_audit


SCHEMA_VERSION = 1
SOURCE_FILES = ("train_cpt.py", "model_backends.py", "token_batches.py", "compiled_adamw.py",
                "native_ops.py", "native_chunked_ce.py", "cpt.py", "packed_data.py", "cpt_orders.py",
                "nonfinite_retry.py", "source_migration.py", "standard_native.py", "lr_intervention.py")


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return value


def load_backend(path):
    config = json.loads(path.read_text())
    required = {"schema_version", "validated", "model", "revision", "sequence_length",
                "effective_batch", "micro_batch", "mask_policy", "fused_optimizer", "options"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Backend configuration is missing {sorted(missing)}")
    if config["schema_version"] != SCHEMA_VERSION or config["validated"] is not True:
        raise ValueError("Backend configuration must have schema_version=1 and validated=true")
    for name in ("sequence_length", "effective_batch", "micro_batch"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Backend {name} must be a positive integer")
    if config["sequence_length"] < 2 or config["micro_batch"] > config["effective_batch"]:
        raise ValueError("Sequence length must be >=2 and micro batch cannot exceed effective batch")
    if config["mask_policy"] not in ("ones", "causal") or not isinstance(config["fused_optimizer"], bool):
        raise ValueError("Expected mask_policy=ones/causal and boolean fused_optimizer")
    compiled_optimizer = config.get("compiled_optimizer", False)
    if not isinstance(compiled_optimizer, bool):
        raise ValueError("Expected boolean compiled_optimizer when provided")
    if compiled_optimizer and config["fused_optimizer"]:
        raise ValueError("compiled_optimizer and fused_optimizer are mutually exclusive")
    if not isinstance(config["options"], dict):
        raise ValueError("Backend options must be a dictionary")
    options = dict(config["options"])
    expected_shape = [config["micro_batch"], config["sequence_length"]]
    if "compile_shape" in options and options["compile_shape"] != expected_shape:
        raise ValueError("Backend compile_shape differs from the validated micro batch/context")
    options["compile_shape"] = expected_shape
    config["options"] = options
    if config.get("execution_mode", "configured") not in ("configured", "standard_native"):
        raise ValueError("Unknown backend execution_mode")
    if config.get("execution_mode") == "standard_native":
        check_native_config(config)
    return config


def make_schedule(total_steps, peak_lr, warmup_steps=None, lr_schedule="multistep"):
    if lr_schedule not in ("multistep", "constant"):
        raise ValueError("Unknown learning-rate schedule")
    if warmup_steps is None:
        warmup_steps = max(1, math.ceil(total_steps * .05))
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= total_steps:
        raise ValueError("Warmup steps must be an integer between zero and the total schedule length")
    schedule = {"total_steps": total_steps, "peak_lr": peak_lr,
            "warmup_steps": warmup_steps,
            "decay_80_step": math.ceil(total_steps * .8),
            "decay_90_step": math.ceil(total_steps * .9),
            "decay_80_factor": .316, "decay_90_factor": .1}
    if lr_schedule == "constant":
        # Keep the historical boundaries for report consumers; no LR decay.
        schedule.update(lr_schedule="constant", decay_80_factor=1.0, decay_90_factor=1.0)
    return schedule


def learning_rate(schedule, step):
    """LR for a zero-based data update, identical before and after resuming."""
    if not 0 <= step < schedule["total_steps"]:
        raise ValueError("Step lies outside the saved schedule")
    if step < schedule["warmup_steps"]:
        factor = (step + 1) / schedule["warmup_steps"]
    elif step < schedule["decay_80_step"]:
        factor = 1.0
    elif step < schedule["decay_90_step"]:
        factor = schedule["decay_80_factor"]
    else:
        factor = schedule["decay_90_factor"]
    return schedule["peak_lr"] * factor


def effective_learning_rate(state, step, input_tokens):
    policy = state.get("lr_intervention")
    if policy is not None and input_tokens >= policy["start_input_tokens"]:
        return intervention_lr(policy, input_tokens)
    return learning_rate(state["schedule"], step)


def configure_lr_intervention(state, path, checkpoint, output, expected_inputs):
    """Apply a future-only policy once; later resumes inherit the saved audit."""
    validate_intervention_audit(state)
    if path is not None:
        policy = validate_intervention(json.loads(path.read_text()))
        if policy is None:
            raise ValueError("An intervention file must contain a policy")
        existing = state.get("lr_intervention")
        if existing is not None:
            if policy != existing:
                raise ValueError("Cannot replace a saved LR intervention when resuming")
        else:
            if checkpoint is None:
                raise ValueError("An LR intervention requires an incomplete resume checkpoint")
            if state["trained_tokens"] > policy["start_input_tokens"]:
                raise ValueError("An LR intervention cannot change the completed training prefix")
            if policy["end_input_tokens"] > expected_inputs:
                raise ValueError("An LR intervention exceeds the configured token budget")
            if policy["start_lr"] != state["schedule"]["peak_lr"]:
                raise ValueError("Intervention start LR must match the original peak LR")
            if policy["end_lr"] > policy["start_lr"]:
                raise ValueError("This intervention supports decreasing LR only")
            state["lr_intervention"] = policy
            state["lr_intervention_audit"] = {
                "policy": policy,
                "applied_at_completed_steps": state["completed_steps"],
                "applied_at_input_tokens": state["trained_tokens"],
                "resumed_checkpoint": str(checkpoint.resolve()),
                "saved_utc": datetime.now(timezone.utc).isoformat(),
                "original_schedule": dict(state["schedule"]),
                "reason": f"Explicit user-requested linear LR intervention: {policy['start_lr']} to {policy['end_lr']} between {policy['start_input_tokens']} and {policy['end_input_tokens']} cumulative input tokens; hold after the end boundary.",
                "policy_file_sha256": cpt.sha(path),
                "lr_counting": "Cumulative input tokens before the optimizer update",
                "authorization": "Explicit --lr-intervention-file; source migration is audited separately",
            }
        validate_intervention_audit(state)
    if state.get("lr_intervention") is not None:
        cpt.write(output / "lr_intervention.json", state["lr_intervention"])
        cpt.write(output / "lr_intervention_audit.json", state["lr_intervention_audit"])


def selected_token_counts(dataset, order, total_steps, effective_batch):
    """Count exact multi-epoch inputs/targets without scanning large prefixes."""
    return prefix_counts(order, total_steps, effective_batch, dataset.sequence_length, len(dataset.tokens))[:2]


def validate_order(order, dataset_length):
    if not isinstance(order, list) or len(order) != dataset_length:
        raise ValueError("Saved shuffle order does not cover this dataset")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in order):
        raise ValueError("Saved shuffle order contains noninteger indices")
    validate_permutation(order, dataset_length)


def validate_resume(state, signature, dataset, args, orders=None):
    if state.get("schema_version") != SCHEMA_VERSION or not isinstance(state.get("signature"), dict):
        raise ValueError("Resume configuration/data/LR/seed differs from the saved training state")
    if getattr(args, "allow_native_backend_migration", False):
        before, after = native_configs(args.resume, args.backend_config, state["signature"], signature, cpt.sha)
        authorize_native_signature(state["signature"], signature, before, after)
    else:
        authorize_signature(state["signature"], signature, getattr(args, "allow_source_migration", False))
    validate_lineage(state)
    if orders is None:
        validate_order(state["order"], len(dataset))
        orders = state["order"]
    if isinstance(orders, EpochOrders):
        if (orders.blocks != len(dataset) or orders.epochs != args.epochs
                or orders.batch != signature["effective_batch"]
                or orders.metadata.get("seed") != args.seed):
            raise ValueError("Saved shuffle metadata differs from dataset/epochs/batch/seed")
    elif args.epochs != 1:
        raise ValueError("Historical list-order checkpoints support one epoch only")
    schedule = state["schedule"]
    epoch_steps = math.ceil(len(dataset) / signature["effective_batch"])
    if not 1 <= schedule["total_steps"] <= epoch_steps * args.epochs:
        raise ValueError("Saved schedule exceeds its epoch budget")
    if args.steps is not None and min(args.steps, epoch_steps * args.epochs) != schedule["total_steps"]:
        raise ValueError("--steps cannot change a resumed LR schedule")
    if schedule != make_schedule(schedule["total_steps"], args.lr,
                                 getattr(args, "warmup_steps", None), getattr(args, "lr_schedule", "multistep")):
        raise ValueError("Saved learning-rate schedule is inconsistent")
    validate_intervention_audit(state)
    audit = state.get("lr_intervention_audit")
    if audit is not None:
        policy = state["lr_intervention"]
        configured_inputs = prefix_counts(orders, schedule["total_steps"], signature["effective_batch"],
                                          dataset.sequence_length, len(dataset.tokens))[0]
        if policy["end_input_tokens"] > configured_inputs or policy["start_lr"] != schedule["peak_lr"] or policy["end_lr"] > policy["start_lr"]:
            raise ValueError("Saved LR intervention differs from the original peak LR or token budget")
        audit_inputs = prefix_counts(orders, audit["applied_at_completed_steps"], signature["effective_batch"],
                                    dataset.sequence_length, len(dataset.tokens))[0]
        if audit_inputs != audit["applied_at_input_tokens"]:
            raise ValueError("LR intervention audit does not match the saved shuffle prefix")
    step = state["completed_steps"]
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < schedule["total_steps"]:
        raise ValueError("Resume requires an incomplete checkpoint")
    inputs, targets, updates = prefix_counts(orders, step, signature["effective_batch"], dataset.sequence_length, len(dataset.tokens))
    if (state["trained_tokens"], state["supervised_tokens"], state["optimizer_updates"]) != (inputs, targets, updates):
        raise ValueError("Saved counters do not match the completed shuffle-order prefix")
    if "epoch_cursor" in state and state["epoch_cursor"] != epoch_cursor(step, epoch_steps):
        raise ValueError("Saved epoch cursor differs from the completed global step")


def epoch_cursor(completed_steps, epoch_steps):
    epoch, local_step = divmod(completed_steps, epoch_steps)
    return {"completed_epochs": epoch, "step_in_epoch": local_step}


def has_full_compile_batch(indices, dataset, micro_batch, label_alias):
    """Avoid calling an eager masked tail the first compiled-shape update."""
    masked_tail = len(dataset.tokens) % dataset.sequence_length != 0 and label_alias == "same_as_input"
    for start in range(0, len(indices), micro_batch):
        chunk = indices[start:start + micro_batch]
        if len(chunk) == micro_batch and not (masked_tail and len(dataset) - 1 in chunk):
            return True
    return False


def prune_periodic_checkpoints(output, keep, signature):
    """Delete only verified, complete periodic checkpoints in this invocation."""
    candidates = []
    for path in output.iterdir():
        match = re.fullmatch(r"checkpoint_step_(\d{8,})", path.name)
        if match and path.is_dir() and not path.is_symlink():
            manifest = path / "checkpoint_manifest.json"
            if manifest.is_file() and (path / "training_state.pt").is_file():
                saved = json.loads(manifest.read_text())
                if saved.get("signature") == signature and saved.get("completed_steps") == int(match[1]):
                    candidates.append((int(match[1]), path))
    removed = []
    for _, path in sorted(candidates)[:-keep]:
        shutil.rmtree(path)
        removed.append(str(path.resolve()))
    return removed


def verify_data(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("tokenizer") != config["model"] or manifest.get("tokenizer_revision") != config["revision"]:
        raise ValueError("Prepared-data tokenizer does not match the selected base model/revision")
    # This trainer owns the block layout: it never passes reset position IDs,
    # document isolation masks, or leading/interior padding to the model.
    if manifest.get("packing") != "document text + EOS, concatenated within each split; no cross-document attention mask":
        raise ValueError("This trainer requires continuous-position, concatenated causal token blocks")
    hashes = {}
    for name in ("train_tokens.npy", "validation_tokens.npy"):
        hashes[name] = cpt.sha(directory / name)
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if expected is not None and hashes[name] != expected:
            raise ValueError(f"Prepared token file checksum mismatch: {name}")
    return manifest, hashes


def check_model_safety(model, native_standard=False):
    if model.config.model_type == "qwen2":
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("Selected backend requires Qwen2 full causal attention")
        if any(layer.attention_type != "full_attention" for layer in model.model.layers):
            raise ValueError("Resolved causal mask cannot be used for sliding/other attention layers")
    elif model.config.model_type == "llama" and native_standard:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
        if (not isinstance(model, LlamaForCausalLM) or not isinstance(model.model, LlamaModel)
                or any(not isinstance(layer, LlamaDecoderLayer) for layer in model.model.layers)
                or getattr(model.config, "use_sliding_window", False)
                or getattr(model.config, "sliding_window", None) not in (None, 0)
                or model.config._attn_implementation != "sdpa"):
            raise ValueError("Native Llama training requires unpatched HF Llama with full causal SDPA attention")
    else:
        raise ValueError("Selected backend requires Qwen2 full causal attention; Llama is supported in standard_native mode only")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("Expected full-parameter training with FP32 master parameters")


def check_optimizer_precision(model, optimizer):
    if not all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
               for p in model.parameters()):
        raise RuntimeError("Master parameter/gradient precision changed")
    for state in optimizer.state.values():
        if state["exp_avg"].dtype != torch.float32 or state["exp_avg_sq"].dtype != torch.float32:
            raise RuntimeError("AdamW moments must remain FP32")


def restore_optimizer(optimizer, saved):
    # Checkpoint tensors were loaded on CPU. Stage the large moment tensors
    # through pinned memory just as cpt.load_model does for master weights;
    # keep foreach Adam's scalar step on CPU unless the optimizer requires GPU.
    step_on_gpu = any(group.get("capturable", False) or group.get("fused", False)
                      for group in optimizer.param_groups)
    for state in saved["state"].values():
        for name, tensor in list(state.items()):
            if isinstance(tensor, torch.Tensor) and (name != "step" or step_on_gpu):
                state[name] = tensor.pin_memory().to("cuda", non_blocking=True)
    # load_state_dict replaces the constructor's parameter-group values.  A
    # saved tensor LR was mapped to CPU alongside the moments and must return
    # to CUDA before a capturable optimizer can read/update it in place.
    if step_on_gpu:
        for group in saved["param_groups"]:
            if isinstance(group.get("lr"), torch.Tensor):
                group["lr"] = group["lr"].to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    optimizer.load_state_dict(saved)


@torch.no_grad()
def evaluate(model, dataset, provider, micro_batch=4, indices=None, native_standard=False):
    """Native, full-block held-out text loss; never compile a validation graph."""
    was_training = model.training
    model.eval()
    total = torch.zeros((), device="cuda")
    count = 0
    selected = range(len(dataset)) if indices is None else indices
    try:
        with preserve_rng(next(model.parameters()).device), (nullcontext() if native_standard else eager_native(model)):
            for start in range(0, len(selected), micro_batch):
                values, targets, _ = provider(dataset, selected[start:min(start + micro_batch, len(selected))])
                if indices is not None:
                    values["attention_mask"] = torch.ones_like(values["input_ids"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**values, use_cache=False).loss
                total += loss * targets
                count += targets
        torch.cuda.synchronize()
        if count == 0:
            raise ValueError("Validation contains no complete supervised token block")
        loss = float(total / count)
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite native validation loss")
        return {"loss": loss, "perplexity": math.exp(loss), "supervised_tokens": count,
                "input_tokens": len(selected) * dataset.sequence_length, "blocks": len(selected),
                "loss_backend": "native eager", "attention_mask": "standard 2D all-ones tensor" if native_standard or indices is not None else "cached all-ones tensor"}
    finally:
        model.train(was_training)


def update(model, forward, optimizer, dataset, provider, indices, micro_batch, mask_policy,
           step_callable=None, retry_native_on_nonfinite=False, step=None, retry_event_callback=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [provider(dataset, indices[start:start + micro_batch])
               for start in range(0, len(indices), micro_batch)]
    targets = sum(n for _, n, _ in batches)
    inputs = sum(n for _, _, n in batches)
    loss_sum = torch.zeros((), device="cuda")
    # A tail of one real token has no next-token label. Count its input while
    # skipping an all-ignored microbatch, rather than asking CE for a NaN mean.
    if targets == 0:
        return loss_sum, loss_sum.clone(), inputs, targets, False
    for values, _, _ in batches:
        if mask_policy == "causal":
            values["attention_mask"] = {"full_attention": None}

    def backward(callable_forward):
        total = torch.zeros((), device="cuda")
        for values, count, _ in batches:
            if count == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = callable_forward(**values, use_cache=False).loss
            weighted = loss * (count / targets)
            weighted.backward()
            total += weighted.detach()
        return total

    def native_backward():
        # Retain the same BF16 autocast, FP32 masters, batch tensors, masks and
        # weighting. Only bypass the configured patches/compiled model calls.
        with eager_native(model):
            return backward(model)

    loss_sum, norm = checked_optimizer_update(
        model, optimizer, lambda: backward(forward), native_backward=native_backward,
        retry_native=retry_native_on_nonfinite, step_callable=step_callable,
        step=step, event_callback=retry_event_callback)
    return loss_sum.detach(), norm.detach(), inputs, targets, True


def save_checkpoint(path, model, tokenizer, optimizer, state, metadata, orders=None):
    """Publish a complete checkpoint directory via one same-filesystem rename."""
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint output already exists: {path}")
    temporary.mkdir()
    if isinstance(orders, EpochOrders):
        orders.copy_to(temporary / "orders")
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    payload = dict(state, schema_version=SCHEMA_VERSION, optimizer=optimizer.state_dict(),
                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    torch.save(payload, temporary / "training_state.pt")
    cpt.write(temporary / "checkpoint_manifest.json", {
        "schema_version": SCHEMA_VERSION, "completed_steps": state["completed_steps"],
        "optimizer_updates": state["optimizer_updates"], "trained_tokens": state["trained_tokens"],
        "supervised_tokens": state["supervised_tokens"], "signature": state["signature"],
        "original_signature": state.get("original_signature", state["signature"]),
        "source_migrations": state.get("source_migrations", []),
        "native_retry_count": state.get("native_retry_count", 0),
        "train_probe": state.get("train_probe"), "train_probe_history": state.get("train_probe_history", []),
        "native_optimizer_adaptations": state.get("native_optimizer_adaptations", []),
        "schedule": state["schedule"], "backend": metadata,
        "lr_intervention": state.get("lr_intervention"),
        "lr_intervention_audit": state.get("lr_intervention_audit"),
        "order_metadata": state.get("order_metadata"), "epoch_cursor": state.get("epoch_cursor"),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "files": {item.name: item.stat().st_size for item in temporary.iterdir() if item.is_file()},
        "verification": "Atomic complete-directory publication; full tensor comparison is a separate diagnostic",
    })
    temporary.rename(path)
    return path.resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend-config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=positive_integer, help="Cap the fixed global schedule to at most --epochs passes")
    p.add_argument("--epochs", type=positive_integer, default=1, help="Fresh deterministic shuffle each epoch; one global LR plan")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=nonnegative_integer,
                   help="Fixed global warmup updates; default retains the historical 5%% warmup")
    p.add_argument("--lr-schedule", choices=("multistep", "constant"), default="multistep",
                   help="After warmup, use historical 80%%/90%% decays or hold the peak LR")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=positive_integer, default=20)
    p.add_argument("--checkpoint-every", type=positive_integer)
    p.add_argument("--keep-checkpoints", type=positive_integer, default=3,
                   help="Retain this run's latest N periodic checkpoints; keep epoch/final/stopped checkpoints")
    p.add_argument("--eval-every", type=positive_integer,
                   help="Native validation every N global steps; default retains historical midpoint evaluation")
    p.add_argument("--resume", type=Path, help="Incomplete checkpoint directory; use a new output directory")
    p.add_argument("--allow-source-migration", action="store_true",
                   help="Resume with changed source_sha256 only; preserve and audit original signature")
    p.add_argument("--lr-intervention-file", type=Path,
                   help="Explicit future-only input-token LR policy; saved separately from the original schedule")
    p.add_argument("--allow-native-backend-migration", action="store_true",
                   help="Audited configured-to-standard-native implementation migration; preserve math/data/order/LR")
    p.add_argument("--train-probe-blocks", type=int, default=0, help="Fixed complete training-corpus blocks for native evaluation; 0 disables")
    p.add_argument("--train-probe-seed", type=int, default=20261009)
    p.add_argument("--retry-native-on-nonfinite", action="store_true",
                   help="Before optimizer mutation, retry a nonfinite update once with the same batch/RNG in native BF16")
    p.add_argument("--stop-after", type=positive_integer, help="Stop at this global step without shortening the schedule")
    p.add_argument("--max-seconds", type=float, help="Wall time checked at update boundaries; saving can extend beyond it")
    return p


def main():
    started_wall = time.perf_counter()
    args = parser().parse_args()
    if args.allow_source_migration and args.allow_native_backend_migration:
        raise ValueError("Use exactly one explicit migration authorization")
    if args.allow_native_backend_migration and args.resume is None:
        raise ValueError("--allow-native-backend-migration requires --resume")
    if args.train_probe_blocks < 0:
        raise ValueError("--train-probe-blocks must be nonnegative")
    if args.allow_source_migration and args.resume is None:
        raise ValueError("--allow-source-migration requires --resume")
    if args.lr_intervention_file is not None and args.resume is None:
        raise ValueError("--lr-intervention-file requires --resume")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    config = load_backend(args.backend_config)
    native_standard = config.get("execution_mode") == "standard_native"
    if args.allow_native_backend_migration and not native_standard:
        raise ValueError("Native backend migration requires standard_native config")
    if native_standard and args.retry_native_on_nonfinite:
        raise ValueError("Standard native execution has strict failures and no native retry")
    data_manifest, data_hashes = verify_data(args.data, config)
    config_hash = cpt.sha(args.backend_config)
    signature = {"backend_config_sha256": config_hash,
                 "data_manifest_sha256": cpt.sha(args.data / "manifest.json"),
                 "token_file_sha256": data_hashes, "peak_lr": args.lr, "seed": args.seed,
                 "sequence_length": config["sequence_length"], "effective_batch": config["effective_batch"],
                 "epochs": args.epochs}
    # Default options retain the exact historical signature/schema.
    if args.warmup_steps is not None:
        signature["warmup_steps"] = args.warmup_steps
    if args.lr_schedule != "multistep":
        signature["lr_schedule"] = args.lr_schedule
    source_directory = Path(__file__).resolve().parent
    source_hashes = {name: cpt.sha(source_directory / name) for name in SOURCE_FILES}
    # Source changes need explicit authorization and a continuous audited
    # source-only signature chain; config/data/LR/seed/epochs stay identical.
    signature["source_sha256"] = source_hashes
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output / "source"
    snapshots.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(source_directory / name, snapshots / name)
    shutil.copy2(args.backend_config, args.output / "backend_config.json")
    shutil.copy2(args.data / "manifest.json", args.output / "data_manifest.json")
    manifest = {"schema_version": SCHEMA_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "backend_config": config, "signature": signature, "source_sha256": source_hashes,
                "retry_native_on_nonfinite": args.retry_native_on_nonfinite,
                "precision": "BF16 autocast; FP32 parameters, gradients and AdamW moments",
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "liger-kernel", "numpy")},
                "timing": "Training-loop totals include data/H2D, synchronization and logging; exclude validation/checkpoint sections. Per-record seconds time the update itself.",
                "stop_boundary": "Wall-clock limits and SIGTERM/SIGINT are checked between updates; final validation/save may extend beyond the limit",
                "shuffle_policy": "Independent sequential CPU-generator permutations per epoch, compact checksummed .npy; partial batches end at epoch boundaries",
                "checkpoint_retention": "Keep latest N periodic checkpoints created in this output; preserve epoch/final/stopped and resume input"}
    cpt.write(args.output / "manifest.json", manifest)
    telemetry = cpt.Telemetry(args.output)
    telemetry.thread.start()
    received_signal = []
    original_handlers = {}
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        original_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, lambda number, frame: received_signal.append(number))
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA GPU with BF16 support is required")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        tokenizer_source = str(args.resume) if args.resume is not None else config["model"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=None if args.resume else config["revision"],
                                                  local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("A tokenizer EOS token is required for trailing block padding")
        train = PackedTokens(args.data, sequence_length=config["sequence_length"], include_tail=True,
                             pad_token_id=tokenizer.eos_token_id)
        validation = PackedTokens(args.data, "validation", sequence_length=config["sequence_length"])
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("Training and complete-block validation data must be nonempty")
        train.explicit_mask = config["mask_policy"] == "ones"
        validation.explicit_mask = True
        provider = cpt.batch if native_standard else PackedTokenBatches()
        epoch_steps = math.ceil(len(train) / config["effective_batch"])
        if args.resume is not None:
            resumed = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
            orders = load_saved_orders(resumed, args.resume)
            validate_resume(resumed, signature, train, args, orders=orders)
            if isinstance(orders, EpochOrders):
                orders.copy_to(args.output / "orders")
                orders = EpochOrders(args.output / "orders", orders.metadata, validate=False)
            state = {key: value for key, value in resumed.items()
                     if key not in ("optimizer", "cpu_rng", "cuda_rng", "schema_version")}
            if args.allow_native_backend_migration:
                apply_native_migration(state, signature, args.resume, args.output, cpt.sha)
            elif args.allow_source_migration:
                apply_source_migration(state, signature, args.resume, args.output)
            else:
                state.setdefault("original_signature", state["signature"])
                state.setdefault("source_migrations", [])
            verify_lineage_snapshots(state, cpt.sha)
            # Periodic checkpoints capture an invocation before its final
            # boundary is written.  Close that history at the resumed prefix,
            # before appending this process's new invocation below.
            if state.get("invocations") and "end_step" not in state["invocations"][-1]:
                state["invocations"][-1]["end_step"] = state["completed_steps"]
        else:
            total_steps = epoch_steps * args.epochs
            if args.steps is not None:
                total_steps = min(args.steps, total_steps)
            orders = EpochOrders.create(args.output / "orders", len(train), config["effective_batch"], args.epochs, args.seed)
            state = {"signature": signature, "order_metadata": orders.metadata,
                     "original_signature": signature, "source_migrations": [],
                     "schedule": make_schedule(total_steps, args.lr, args.warmup_steps, args.lr_schedule), "completed_steps": 0,
                     "optimizer_updates": 0, "trained_tokens": 0, "supervised_tokens": 0,
                     "train_seconds": 0., "steady_seconds": 0., "steady_tokens": 0,
                     "validation_history": [], "validation_before": None,
                     "invocations": []}
        state.setdefault("native_retry_count", 0)
        state["train_probe"] = prepare_train_probe(train, args.train_probe_blocks, args.train_probe_seed, state.get("train_probe"))
        state.setdefault("train_probe_history", [])
        if state["train_probe"] is not None:
            cpt.write(args.output / "train_probe.json", state["train_probe"])
        state["epoch_cursor"] = epoch_cursor(state["completed_steps"], epoch_steps)
        schedule = state["schedule"]
        if args.stop_after is not None and args.stop_after <= state["completed_steps"]:
            raise ValueError("--stop-after must exceed the resumed completed-step boundary")
        expected_inputs, expected_targets = selected_token_counts(train, orders, schedule["total_steps"], config["effective_batch"])
        configure_lr_intervention(state, args.lr_intervention_file, args.resume, args.output, expected_inputs)
        manifest.update(schedule=schedule, configured_input_tokens=expected_inputs,
                        configured_supervised_tokens=expected_targets, dataset_input_tokens=len(train.tokens),
                        epochs=args.epochs, epoch_steps=epoch_steps, order_metadata=state.get("order_metadata"),
                        original_signature=state["original_signature"], source_migrations=state["source_migrations"],
                        train_probe=state["train_probe"], execution_mode="standard_native" if native_standard else "configured",
                        resumed_completed_steps=state["completed_steps"], resumed_input_tokens=state["trained_tokens"],
                        lr_intervention=state.get("lr_intervention"), lr_intervention_audit=state.get("lr_intervention_audit"))
        cpt.write(args.output / "manifest.json", manifest)
        model_args = argparse.Namespace(model=str(args.resume) if args.resume else config["model"],
                                       revision=None if args.resume else config["revision"])
        model = load_native_model(model_args) if native_standard else cpt.load_model(model_args)
        check_model_safety(model, native_standard=native_standard)
        optimizer_compile = None
        step_callable = None
        if native_standard:
            forward = model
            backend_metadata = {"execution_mode": "standard_native", "kernels": "native HF",
                                "loss": "native HF causal LM", "compile_scope": "none", "optimizer_compile": None,
                                "attention_mask": "standard 2D all-ones", "labels": "independent stacked clones",
                                "data_provider": "original cpt.batch", "optimizer": "standard uncompiled torch.optim.AdamW foreach",
                                "native_configure_model_called": False}
            optimizer = make_native_optimizer(model, args.lr)
        else:
            forward, backend_metadata = configure_model(model, config["options"])
            if config.get("compiled_optimizer", False):
                optimizer, step_callable, optimizer_compile = make_compiled_adamw(model.parameters(), args.lr)
            else:
                optimizer = cpt.make_optimizer(model, args.lr, config["fused_optimizer"])
        backend_metadata["optimizer_compile"] = optimizer_compile
        state["optimizer_compile"] = optimizer_compile
        if args.resume is not None:
            if native_standard:
                adaptation = dict(checkpoint=str(args.resume.resolve()), completed_steps=state["completed_steps"],
                                  **adapt_native_optimizer(resumed["optimizer"]))
                state.setdefault("native_optimizer_adaptations", []).append(adaptation)
                cpt.write(args.output / "native_optimizer_adaptation.json", adaptation)
            restore_optimizer(optimizer, resumed["optimizer"])
            check_optimizer_precision(model, optimizer)
            torch.set_rng_state(resumed["cpu_rng"])
            torch.cuda.set_rng_state(resumed["cuda_rng"])
            del resumed
            gc.collect()
        telemetry.phase = "validation_before"
        before = evaluate(model, validation, provider, native_standard=native_standard)
        if state["validation_before"] is None:
            state["validation_before"] = before
        cpt.write(args.output / "validation_before.json", dict(step=state["completed_steps"], **before))
        print(json.dumps({"event": "validation_before", "step": state["completed_steps"], **before}), flush=True)
        def record_train_probe():
            if state["train_probe"] is None:
                return
            metrics = dict(step=state["completed_steps"], cumulative_tokens=state["trained_tokens"],
                           **evaluate(model, train, cpt.batch, indices=state["train_probe"]["indices"], native_standard=native_standard))
            state["train_probe_history"].append(metrics)
            cpt.write(args.output / "train_probe_history.json", state["train_probe_history"])
            print(json.dumps({"event": "train_probe", **metrics}), flush=True)

        # Probe uses ordinary all-ones masks even when the configured training
        # backend has a resolved causal mask; evaluation restores all RNG.
        train.explicit_mask = True if native_standard else train.explicit_mask
        record_train_probe()
        invocation = {"output": str(args.output.resolve()), "start_step": state["completed_steps"],
                      "first_full_shape_update_seconds_including_compile": None, "train_seconds": 0.,
                      "input_tokens": 0, "steady_seconds": 0., "steady_tokens": 0,
                      "optimizer_compile": optimizer_compile,
                      "retry_native_on_nonfinite": args.retry_native_on_nonfinite,
                      "native_retry_count": 0}
        state["invocations"].append(invocation)
        first_full_shape_seen = False
        stop_reason = None
        torch.cuda.reset_peak_memory_stats()
        with (args.output / "training.jsonl").open("x") as log, (args.output / "native_retries.jsonl").open("x") as retry_log:
            while state["completed_steps"] < schedule["total_steps"]:
                if received_signal:
                    stop_reason = signal.Signals(received_signal[0]).name
                    break
                if args.stop_after is not None and state["completed_steps"] >= args.stop_after:
                    stop_reason = "stop_after"
                    break
                if args.max_seconds is not None and time.perf_counter() - started_wall >= args.max_seconds:
                    stop_reason = "max_seconds"
                    break
                loop_begin = time.perf_counter()
                step = state["completed_steps"]
                indices = order_indices(orders, step, config["effective_batch"])
                lr_input_tokens = state["trained_tokens"]
                rate = effective_learning_rate(state, step, lr_input_tokens)
                if native_standard:
                    set_native_lr(optimizer, rate)
                elif config.get("compiled_optimizer", False):
                    set_lr(optimizer, rate)
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                first_full = not first_full_shape_seen and has_full_compile_batch(
                    indices, train, config["micro_batch"], config["options"].get("compile_label_alias", "any"))
                telemetry.phase = "training_initialization" if first_full else "training"
                torch.cuda.synchronize()
                begin = time.perf_counter()
                retry_events = []

                def record_retry(event):
                    retry_events.append(event)
                    retry_log.write(json.dumps(event, allow_nan=False) + "\n")
                    retry_log.flush()
                    print(json.dumps(event, allow_nan=False), flush=True)

                if native_standard:
                    loss, norm, inputs, targets, applied = update_native(model, optimizer, train, provider, indices, config["micro_batch"])
                else:
                    loss, norm, inputs, targets, applied = update(model, forward, optimizer, train, provider, indices,
                                                                 config["micro_batch"], config["mask_policy"],
                                                                 step_callable=step_callable,
                                                                 retry_native_on_nonfinite=args.retry_native_on_nonfinite,
                                                                 step=step + 1, retry_event_callback=record_retry)
                torch.cuda.synchronize()
                duration = time.perf_counter() - begin
                if first_full:
                    first_full_shape_seen = True
                    invocation["first_full_shape_update_seconds_including_compile"] = duration
                    check_optimizer_precision(model, optimizer)
                    print(json.dumps({"event": "first_full_shape_update", "seconds_including_compile": duration,
                                      "compile_scope": backend_metadata["compile_scope"]}), flush=True)
                state["completed_steps"] += 1
                state["epoch_cursor"] = epoch_cursor(state["completed_steps"], epoch_steps)
                state["optimizer_updates"] += int(applied)
                if retry_events:
                    state["native_retry_count"] += 1
                    invocation["native_retry_count"] += 1
                state["trained_tokens"] += inputs
                state["supervised_tokens"] += targets
                invocation["input_tokens"] += inputs
                record = {"step": state["completed_steps"], "loss": float(loss), "grad_norm": float(norm),
                          "epoch": step // epoch_steps + 1, "step_in_epoch": step % epoch_steps + 1,
                          "lr": rate, "effective_fp32_lr": float(torch.tensor(rate, dtype=torch.float32)) if native_standard or config.get("compiled_optimizer", False) else rate,
                          "lr_input_tokens": lr_input_tokens,
                          "input_tokens": inputs, "supervised_tokens": targets,
                          "cumulative_tokens": state["trained_tokens"], "seconds": duration,
                          "native_retry_applied": bool(retry_events),
                          "optimizer_applied": applied, "first_full_shape_update": first_full}
                log.write(json.dumps(record) + "\n")
                if state["completed_steps"] % args.log_every == 0 or state["completed_steps"] == schedule["total_steps"]:
                    log.flush()
                    cpt.write(args.output / "progress.json", record)
                    print(json.dumps({"event": "training", **record}), flush=True)
                loop_duration = time.perf_counter() - loop_begin
                state["train_seconds"] += loop_duration
                invocation["train_seconds"] += loop_duration
                if not first_full:
                    state["steady_seconds"] += loop_duration
                    state["steady_tokens"] += inputs
                    invocation["steady_seconds"] += loop_duration
                    invocation["steady_tokens"] += inputs
                evaluation_due = (state["completed_steps"] % args.eval_every == 0 if args.eval_every is not None
                                  else state["completed_steps"] == max(1, schedule["total_steps"] // 2))
                if evaluation_due and state["completed_steps"] < schedule["total_steps"]:
                    telemetry.phase = "validation"
                    metrics = dict(step=state["completed_steps"], **evaluate(model, validation, provider, native_standard=native_standard))
                    state["validation_history"].append(metrics)
                    cpt.write(args.output / "validation_history.json", state["validation_history"])
                    print(json.dumps({"event": "validation", **metrics}), flush=True)
                    record_train_probe()
                epoch_boundary = args.epochs > 1 and state["completed_steps"] % epoch_steps == 0
                periodic_due = args.checkpoint_every is not None and state["completed_steps"] % args.checkpoint_every == 0
                if (epoch_boundary or periodic_due) and state["completed_steps"] < schedule["total_steps"]:
                    telemetry.phase = "saving"
                    log.flush()
                    name = (f"checkpoint_epoch_{state['completed_steps'] // epoch_steps:04d}" if epoch_boundary
                            else f"checkpoint_step_{state['completed_steps']:08d}")
                    saved = save_checkpoint(args.output / name, model, tokenizer, optimizer, state, backend_metadata, orders)
                    cpt.write(args.output / "latest_checkpoint.json", {"path": str(saved), "step": state["completed_steps"]})
                    print(json.dumps({"event": "checkpoint", "path": str(saved), "step": state["completed_steps"]}), flush=True)
                    removed = prune_periodic_checkpoints(args.output, args.keep_checkpoints, signature)
                    if removed:
                        print(json.dumps({"event": "checkpoint_retention", "removed": removed}), flush=True)
        complete = state["completed_steps"] == schedule["total_steps"]
        check_optimizer_precision(model, optimizer)
        actual_inputs, actual_targets, actual_updates = prefix_counts(orders, state["completed_steps"], config["effective_batch"],
                                                                     train.sequence_length, len(train.tokens))
        if (state["trained_tokens"], state["supervised_tokens"], state["optimizer_updates"]) != (actual_inputs, actual_targets, actual_updates):
            raise RuntimeError("Training counters differ from the completed shuffled data prefix")
        if complete and state["trained_tokens"] != expected_inputs:
            raise RuntimeError("Completed training did not consume its configured exact token budget")
        telemetry.phase = "validation_final"
        final_metrics = dict(step=state["completed_steps"], **evaluate(model, validation, provider, native_standard=native_standard))
        state["validation_history"].append(final_metrics)
        cpt.write(args.output / "validation_history.json", state["validation_history"])
        print(json.dumps({"event": "validation_final", **final_metrics}), flush=True)
        record_train_probe()
        invocation["end_step"] = state["completed_steps"]
        invocation["stop_reason"] = stop_reason
        telemetry.phase = "saving"
        checkpoint = save_checkpoint(args.output / ("final_checkpoint" if complete else "stopped_checkpoint"),
                                     model, tokenizer, optimizer, state, backend_metadata, orders)
        cpt.write(args.output / "latest_checkpoint.json", {"path": str(checkpoint), "step": state["completed_steps"]})
        result = {"status": "complete" if complete else "stopped", "stop_reason": stop_reason,
                  "completed_steps": state["completed_steps"], "optimizer_updates": state["optimizer_updates"],
                  "configured_steps": schedule["total_steps"], "trained_tokens": state["trained_tokens"],
                  "supervised_tokens": state["supervised_tokens"], "configured_input_tokens": expected_inputs,
                  "dataset_input_tokens": len(train.tokens), "epochs": args.epochs, "epoch_steps": epoch_steps,
                  "epoch_cursor": state["epoch_cursor"], "order_metadata": state.get("order_metadata"),
                  "all_dataset_tokens_consumed": complete and state["trained_tokens"] == len(train.tokens) * args.epochs,
                  "train_seconds_excluding_validation_and_save": state["train_seconds"],
                  "tokens_per_second_including_first_update": state["trained_tokens"] / state["train_seconds"] if state["train_seconds"] else None,
                  "steady_tokens_per_second": state["steady_tokens"] / state["steady_seconds"] if state["steady_seconds"] else None,
                  "steady_tokens": state["steady_tokens"], "steady_seconds": state["steady_seconds"],
                  "steady_definition": "Exclude the first full-shape update in each invocation; loop timings include data/H2D, synchronization and logging, and exclude validation/save sections",
                  "invocations": state["invocations"], "schedule": schedule, "backend": backend_metadata,
                  "lr_intervention": state.get("lr_intervention"), "lr_intervention_audit": state.get("lr_intervention_audit"),
                  "original_signature": state["original_signature"], "source_migrations": state["source_migrations"],
                  "native_retry_count": state["native_retry_count"],
                  "train_probe": state["train_probe"], "train_probe_history": state["train_probe_history"],
                  "native_optimizer_adaptations": state.get("native_optimizer_adaptations", []),
                  "optimizer_compile": optimizer_compile,
                  "validation_before": state["validation_before"], "validation_after": final_metrics,
                  "validation_history": state["validation_history"],
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "gpu_training_this_invocation": telemetry.summary("training"),
                  "checkpoint": str(checkpoint), "full_parameter_training": True,
                  "wall_seconds_this_invocation": time.perf_counter() - started_wall,
                  "checkpoint_verification": "Atomic publication; independent reload/resume verification is separate"}
        cpt.write(args.output / "result.json", result)
        print(json.dumps({"event": "training_complete" if complete else "training_stopped", **result}), flush=True)
    except Exception as error:
        cpt.write(args.output / "result.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        for number, handler in original_handlers.items():
            signal.signal(number, handler)
        telemetry.stop.set()
        telemetry.thread.join(timeout=6)


if __name__ == "__main__":
    main()
