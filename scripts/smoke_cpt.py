"""Bounded BF16 full-parameter CPT smoke, not a mathematical benchmark.

Use --fixture for generated arithmetic documents, or --data for JSONL with text.
Output directories must be new. No existing experiment or environment is modified.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time


def blocks(tokenizer, texts, length):
    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
    return [tokens[i:i + length] for i in range(0, len(tokens) - length + 1, length)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--fixture", action="store_true")
    data.add_argument("--data", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.steps) < 1 or args.warmup_steps < 0:
        parser.error("length, batch size, steps must be positive; warmup must be nonnegative")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; this smoke has not run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU must support BF16")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                             local_files_only=args.local_files_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if args.fixture:
        texts = [f"Arithmetic example: {a} + {b} = {a + b}. "
                 f"Subtracting {b} gives {a + b} - {b} = {a}."
                 for a in range(1, 65) for b in range(1, 17)]
    else:
        # Smoke datasets only: materialize and tokenize before the timed region.
        texts = [json.loads(line)["text"] for line in args.data.read_text().splitlines() if line.strip()]
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Every document must contain nonempty text")
    packed = blocks(tokenizer, texts, args.sequence_length)
    needed = args.batch_size * (args.warmup_steps + args.steps)
    if len(packed) < needed:
        raise ValueError(f"Need {needed} full blocks, found {len(packed)}; supply more text")
    batches = [torch.tensor(packed[i:i + args.batch_size], dtype=torch.long).pin_memory()
               for i in range(0, needed, args.batch_size)]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    resolved_commit = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    parameters = list(model.parameters())
    assert all(p.requires_grad for p in parameters), "All model parameters must be trainable"
    assert all(p.dtype == torch.bfloat16 for p in parameters), "Expected BF16 parameters"
    probe_name, probe = next((n, p) for n, p in model.named_parameters() if "q_proj.weight" in n)
    before = probe.detach().clone()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.1)

    def update(batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = batch.cuda(non_blocking=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach(), norm.detach()

    for batch in batches[:args.warmup_steps]:
        update(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses, norms = [], []
    for batch in batches[args.warmup_steps:]:
        loss, norm = update(batch)
        losses.append(loss)
        norms.append(norm)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    changed = torch.count_nonzero(probe.detach() != before).item()
    if changed == 0:
        raise RuntimeError("Probe weights did not change; inspect BF16 update precision")
    del before

    # Save and reload both model and optimizer; compare a fixed-batch loss.
    model.eval()
    fixed = batches[0].cuda()
    with torch.no_grad():
        saved_loss = model(input_ids=fixed, labels=fixed).loss.float().item()
    checkpoint = args.output / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, checkpoint / "training_state.pt")
    total_steps = args.warmup_steps + args.steps
    # Drop every reference to the original model before loading another copy.
    del optimizer, parameters, probe, model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    del state
    assert all(int(s["step"].item()) == total_steps for s in optimizer.state.values())
    model.eval()
    with torch.no_grad():
        reloaded_loss = model(input_ids=fixed, labels=fixed).loss.float().item()
    assert abs(saved_loss - reloaded_loss) < 1e-5, (saved_loss, reloaded_loss)
    resumed_loss, _ = update(batches[0])
    assert all(int(s["step"].item()) == total_steps + 1 for s in optimizer.state.values())
    result = {
        "status": "passed", "scope": "CPT correctness smoke; short timing is not a throughput benchmark",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resolved_model_commit": resolved_commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "versions": {k: importlib.metadata.version(k) for k in ["torch", "transformers"]},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "probe": probe_name, "probe_changed_elements": changed,
        "losses": [x.float().item() for x in losses], "grad_norms": [x.float().item() for x in norms],
        "timed_seconds": elapsed, "seconds_per_update": elapsed / args.steps,
        "input_tokens_per_second": args.steps * args.batch_size * args.sequence_length / elapsed,
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "checkpoint_loss": saved_loss, "reloaded_loss": reloaded_loss,
        "resumed_loss": resumed_loss.float().item(), "resume_verified": True,
        "optimizer_state_dtype": "PyTorch AdamW defaults; not FP32 master-weight training"
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
