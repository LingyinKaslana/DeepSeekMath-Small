"""Calibrate or train the 10M-token CPT pilot with BF16 AMP and FP32 state.

Calibration always starts each case from the same base weights and uses the same
ordered token blocks and effective batch. Calibration updates are never used by
the training pilot. Full-token validation uses the ordinary HF loss in eval mode.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from types import MethodType

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_data import PackedTokens


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Telemetry:
    def __init__(self, directory):
        self.phase = "loading"
        self.stop = threading.Event()
        self.rows = []
        self.path = directory / "gpu_samples.jsonl"
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with self.path.open("x") as stream:
            while not self.stop.is_set():
                try:
                    output = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], text=True, timeout=5)
                    util, memory, power = [float(v.strip()) for v in output.strip().splitlines()[0].split(",")]
                    row = dict(time=time.time(), phase=self.phase, utilization=util,
                               memory_mib=memory, power_w=power)
                    self.rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                except (subprocess.SubprocessError, ValueError):
                    pass
                self.stop.wait(0.5)

    def summary(self, phase):
        rows = [r for r in self.rows if r["phase"] == phase]
        return {"samples": len(rows), **{key: sum(r[key] for r in rows) / len(rows)
                for key in ("utilization", "memory_mib", "power_w")}} if rows else {"samples": 0}


def load_model(args):
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
    # Stage mmap-backed CPU weights through pinned memory; default pageable H2D
    # transfers were slow during the earlier native BF16 checkpoint round trip.
    model._apply(lambda t: t.pin_memory().to("cuda", non_blocking=True) if t.device.type == "cpu" else t)
    torch.cuda.synchronize()
    model.config.use_cache = False
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    print(json.dumps({"event": "model_loaded", "seconds": time.perf_counter() - start,
                      "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    return model


def make_optimizer(model, lr, fused):
    # Keep decay/betas consistent across every calibration and the pilot.
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=fused, foreach=not fused)


def set_loss(model, original, kind):
    if not hasattr(model, "_cpt_native_loss_function"):
        model._cpt_native_loss_function = model.loss_function
    model.loss_function = model._cpt_native_loss_function
    if kind == "native":
        model.forward = original
    elif kind == "liger_ce":
        from liger_kernel.transformers import LigerCrossEntropyLoss
        model.forward = original

        def cross_entropy(logits, labels, vocab_size, **kwargs):
            shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            return LigerCrossEntropyLoss()(logits.float().reshape(-1, vocab_size), shifted.reshape(-1))

        model.loss_function = cross_entropy
    else:
        from liger_kernel.transformers.model.qwen2 import lce_forward
        model.forward = MethodType(lce_forward, model)


def batch(dataset, indices):
    rows = [dataset[int(i)] for i in indices]
    values = {key: torch.stack([r[key] for r in rows]).pin_memory() for key in rows[0]}
    valid = int((values["labels"][:, 1:] != -100).sum())
    input_tokens = int((values["labels"] != -100).sum())
    if getattr(dataset, "explicit_mask", False):
        # HF 4.55 otherwise treats auto-generated position_ids as potential
        # packed sequences and materializes a 4D mask, blocking native flash.
        # These blocks have continuous positions and causal attention. Tail
        # padding is after real tokens, so it cannot affect their predictions.
        values["attention_mask"] = torch.ones_like(values["input_ids"])
    return {k: v.to("cuda", non_blocking=True) for k, v in values.items()}, valid, input_tokens


def update(model, optimizer, dataset, indices, micro_batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [batch(dataset, indices[i:i + micro_batch]) for i in range(0, len(indices), micro_batch)]
    denominator = sum(n for _, n, _ in batches)
    loss_sum = torch.zeros((), device="cuda")
    for values, n, _ in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**values, use_cache=False).loss
        weighted = loss * (n / denominator)
        weighted.backward()
        loss_sum += weighted.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(loss_sum):
        raise RuntimeError("Nonfinite training loss")
    optimizer.step()
    return loss_sum.detach(), norm.detach(), sum(n for _, _, n in batches)


@torch.no_grad()
def evaluate(model, dataset, micro_batch):
    model.eval()
    previous_loss = model.loss_function
    if hasattr(model, "_cpt_native_loss_function"):
        model.loss_function = model._cpt_native_loss_function
    total, count = torch.zeros((), device="cuda"), 0
    for start in range(0, len(dataset), micro_batch):
        values, n, _ = batch(dataset, range(start, min(start + micro_batch, len(dataset))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**values, use_cache=False).loss
        total += loss * n
        count += n
    torch.cuda.synchronize()
    loss = float(total / count)
    model.loss_function = previous_loss
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite validation loss")
    return {"loss": loss, "perplexity": math.exp(loss), "supervised_tokens": count,
            "blocks": len(dataset)}


def _numerical_check(model, original, dataset):
    values, _, _ = batch(dataset, [0])
    values = {k: v[:, :512] for k, v in values.items()}
    probes = [model.model.layers[0].self_attn.q_proj.weight, model.model.layers[0].mlp.down_proj.weight]
    records = []
    reference = None
    for kind in ("native", "liger_ce", "liger"):
        set_loss(model, original, kind)
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**values).loss
        loss.backward()
        gradients = [p.grad.detach().clone() for p in probes]
        if reference is None:
            reference = (loss.detach().clone(), gradients)
            records.append({"loss_type": kind, "status": "passed", "loss": float(loss.detach())})
        else:
            error = abs(float(loss.detach()) - float(reference[0]))
            relative = [float((g - old).norm() / old.norm().clamp_min(1e-12))
                        for g, old in zip(gradients, reference[1])]
            passed = error < 0.003 and max(relative) < 0.05
            records.append({"loss_type": kind, "status": "passed" if passed else "rejected_numerical_check",
                            "loss": float(loss.detach()),
                            "absolute_loss_difference": error, "probe_gradient_relative_l2": relative})
    model.zero_grad(set_to_none=True)
    set_loss(model, original, "native")
    return records


def numerical_check(model, original, dataset):
    # Compare loss implementations under a stable attention reference. The
    # implementation comparison on memory-efficient attention differed by
    # >10% in the small q-probe; math attention separates those numerical
    # effects from the loss implementation (without relaxing tolerances).
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        records = _numerical_check(model, original, dataset)
    for record in records:
        record["attention_reference"] = "math"
    return records


def calibrate(args, model, tokenizer, telemetry):
    dataset = PackedTokens(args.data, sequence_length=args.sequence_length)
    dataset.explicit_mask = args.attention_mask == "ones"
    original = model.forward
    numerical = numerical_check(model, original, dataset)
    write(args.output / "numerical_check.json", numerical)
    print(json.dumps({"event": "numerical_check", "results": numerical}), flush=True)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cases = []

    def run_case(name, micro, fused, loss_type):
        model.load_state_dict(reference)
        set_loss(model, original, loss_type)
        optimizer = make_optimizer(model, args.lr, fused)
        telemetry.phase = name + "/warmup"
        records = []
        try:
            for step in range(args.warmup + args.steps):
                indices = range(step * args.effective_batch, (step + 1) * args.effective_batch)
                if step == args.warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    telemetry.phase = name
                    started = time.perf_counter()
                loss, norm, tokens = update(model, optimizer, dataset, indices, micro)
                records.append((loss, norm, tokens))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            measured = records[args.warmup:]
            result = dict(name=name, status="passed", micro_batch=micro, effective_batch=args.effective_batch,
                attention_mask=args.attention_mask,
                fused_optimizer=fused, loss_type=loss_type, seconds=seconds,
                seconds_per_update=seconds / args.steps,
                tokens_per_second=sum(r[2] for r in measured) / seconds,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                losses=[float(r[0]) for r in measured], grad_norms=[float(r[1]) for r in measured],
                telemetry=telemetry.summary(name))
            for state in optimizer.state.values():
                assert state["exp_avg"].dtype == torch.float32 and state["exp_avg_sq"].dtype == torch.float32
        except torch.OutOfMemoryError:
            result = dict(name=name, status="out_of_memory", micro_batch=micro, loss_type=loss_type,
                          fused_optimizer=fused)
        finally:
            telemetry.phase = "between_cases"
            optimizer.zero_grad(set_to_none=True)
            del optimizer, records
            gc.collect()
            torch.cuda.empty_cache()
        cases.append(result)
        write(args.output / "cases.json", cases)
        print(json.dumps(result), flush=True)
        return result

    native = [run_case(f"native_m{m}_foreach", m, False, "native")
              for m in (2, 4, 8, 16) if m <= args.effective_batch and args.effective_batch % m == 0]
    passed = [c for c in native if c["status"] == "passed"]
    assert passed, "No baseline case completed"
    best = max(passed, key=lambda c: c["tokens_per_second"])
    run_case(f"native_m{best['micro_batch']}_fused", best["micro_batch"], True, "native")
    for check in numerical:
        if check["loss_type"] == "native" or check["status"] != "passed":
            continue
        kind = check["loss_type"]
        run_case(f"{kind}_m{best['micro_batch']}_fused", best["micro_batch"], True, kind)
        if best["micro_batch"] != args.effective_batch:
            run_case(f"{kind}_m{args.effective_batch}_fused", args.effective_batch, True, kind)
    # Profiling is outside steady-state timing and never affects the pilot.
    model.load_state_dict(reference)
    set_loss(model, original, "native")
    optimizer = make_optimizer(model, args.lr, False)
    telemetry.phase = "profiling_native"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
        update(model, optimizer, dataset, range(args.effective_batch), 2)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output / "native_trace.json"))
    (args.output / "native_profile.txt").write_text(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    best = max((c for c in cases if c["status"] == "passed"), key=lambda c: c["tokens_per_second"])
    result = dict(status="complete", selected=best, cases=cases, numerical_checks=numerical,
        comparison="same initial weights, ordered real data, effective batch, BF16 AMP/FP32 state and learning rate",
        warmup_updates=args.warmup, measured_updates=args.steps,
        pilot_initialized_from="original Coder Base; calibration weights are discarded")
    write(args.output / "result.json", result)
    print(json.dumps({"event": "calibration_complete", "selected": best}), flush=True)


def train(args, model, tokenizer, telemetry):
    calibration_manifest = json.loads((args.calibration.parent / "manifest.json").read_text())
    assert calibration_manifest["data_manifest_sha256"] == sha(args.data / "manifest.json")
    for key in ("model", "revision", "sequence_length", "effective_batch", "attention_mask"):
        assert calibration_manifest["config"][key] == getattr(args, key), key
    selection = json.loads(args.calibration.read_text())["selected"]
    assert selection["status"] == "passed"
    assert selection["effective_batch"] == args.effective_batch
    assert selection["attention_mask"] == args.attention_mask
    original = model.forward
    set_loss(model, original, selection["loss_type"])
    dataset = PackedTokens(args.data, sequence_length=args.sequence_length, include_tail=True,
                           pad_token_id=tokenizer.eos_token_id)
    validation = PackedTokens(args.data, "validation", sequence_length=args.sequence_length)
    dataset.explicit_mask = validation.explicit_mask = args.attention_mask == "ones"
    optimizer = make_optimizer(model, args.lr, selection["fused_optimizer"])
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    total_steps = math.ceil(len(order) / args.effective_batch)
    warmup = max(1, math.ceil(total_steps * 0.05))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < math.ceil(total_steps * 0.8):
            return 1.0
        if step < math.ceil(total_steps * 0.9):
            return 0.316
        return 0.1

    telemetry.phase = "validation_before"
    initial = evaluate(model, validation, 4)
    write(args.output / "validation_before.json", initial)
    print(json.dumps({"event": "validation_before", **initial}), flush=True)
    history, validation_history = [], []
    trained_tokens, seconds = 0, 0.0
    torch.cuda.reset_peak_memory_stats()
    for step, start in enumerate(range(0, len(order), args.effective_batch)):
        telemetry.phase = "training"
        lr = args.lr * lr_factor(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.synchronize()
        begin = time.perf_counter()
        loss, norm, tokens = update(model, optimizer, dataset,
                                   order[start:start + args.effective_batch], selection["micro_batch"])
        torch.cuda.synchronize()
        duration = time.perf_counter() - begin
        seconds += duration
        trained_tokens += tokens
        record = dict(step=step + 1, loss=float(loss), grad_norm=float(norm), lr=lr,
                      input_tokens=tokens, cumulative_tokens=trained_tokens, seconds=duration)
        history.append(record)
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if (step + 1) % 10 == 0 or step + 1 == total_steps:
            print(json.dumps({"event": "training", **record}), flush=True)
            write(args.output / "progress.json", record)
        if step + 1 in {total_steps // 2, total_steps}:
            telemetry.phase = "validation"
            metrics = dict(step=step + 1, **evaluate(model, validation, 4))
            validation_history.append(metrics)
            write(args.output / "validation_history.json", validation_history)
            print(json.dumps({"event": "validation", **metrics}), flush=True)
    assert trained_tokens == len(dataset.tokens), (trained_tokens, len(dataset.tokens))
    telemetry.phase = "saving"
    checkpoint = args.output / "final_checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save(dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), order=order, completed_steps=total_steps,
                    trained_tokens=trained_tokens, lr_schedule="5% linear warmup; 80%->0.316; 90%->0.1",
                    peak_lr=args.lr), checkpoint / "training_state.pt")
    # Independently verify every saved weight and optimizer moment on CPU.
    # Training results use AMP eval before saving; no additional optimizer update.
    from safetensors import safe_open
    checked = 0
    for filename in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                assert torch.equal(tensor, model.state_dict()[name].detach().cpu()), name
                checked += 1
    restored = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    current = optimizer.state_dict()
    assert restored["completed_steps"] == total_steps and restored["trained_tokens"] == trained_tokens
    assert restored["order"] == order and restored["optimizer"]["param_groups"] == current["param_groups"]
    assert set(restored["optimizer"]["state"]) == set(current["state"])
    for index, saved in restored["optimizer"]["state"].items():
        assert int(saved["step"]) == total_steps
        for key in ("exp_avg", "exp_avg_sq"):
            assert saved[key].dtype == torch.float32 and torch.isfinite(saved[key]).all()
            assert torch.equal(saved[key], current["state"][index][key].detach().cpu()), (index, key)
    restored_state_count = len(restored["optimizer"]["state"])
    del restored, current
    result = dict(status="complete", trained_tokens=trained_tokens, optimizer_updates=total_steps,
        train_seconds_excluding_validation_and_save=seconds, tokens_per_second=trained_tokens / seconds,
        validation_before=initial, validation_after=validation_history[-1], validation_history=validation_history,
        selected_backend=selection, peak_lr=args.lr, warmup_steps=warmup,
        precision="BF16 autocast compute; FP32 parameters, gradients and AdamW moments",
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_training=telemetry.summary("training"), checkpoint=str(checkpoint.resolve()),
        saved_weight_tensors_verified=checked, saved_optimizer_states_verified=restored_state_count,
        full_parameter_training=True)
    write(args.output / "result.json", result)
    print(json.dumps({"event": "training_complete", **result}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["calibrate", "train"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("DeepSeekMath-Small/data/processed/openwebmath_10m_s42"))
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--revision", default="df3ce67c0e24480f20468b6ef2894622d69eb73b")
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--attention-mask", choices=["none", "ones"], default="ones")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration", type=Path)
    args = p.parse_args()
    if args.mode == "train" and args.calibration is None:
        p.error("Training requires --calibration pointing to result.json")
    if min(args.sequence_length, args.effective_batch, args.steps, args.lr) <= 0 or args.warmup < 1:
        p.error("Invalid size, learning rate or warmup")
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={name: sha(Path(__file__).parent / name) for name in ["cpt.py", "packed_data.py"]},
        data_manifest_sha256=sha(args.data / "manifest.json"),
        calibration_sha256=sha(args.calibration) if args.calibration else None,
        precision="BF16 autocast; FP32 parameters, gradients and optimizer moments",
        versions={k: importlib.metadata.version(k) for k in ["torch", "transformers", "liger-kernel", "numpy"]}))
    telemetry = Telemetry(args.output)
    telemetry.thread.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
        model = load_model(args)
        if args.mode == "calibrate":
            calibrate(args, model, tokenizer, telemetry)
        else:
            train(args, model, tokenizer, telemetry)
    finally:
        telemetry.stop.set()
        telemetry.thread.join(timeout=6)


if __name__ == "__main__":
    main()
