"""Generate every benchmark answer from a verified CPT checkpoint with native HF."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from math_benchmark_protocol import build_prompt, load_benchmarks, stop_strings


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Smoke only: first N questions in EACH dataset")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0 and args.max_new_tokens > 0
    checkpoint, vendor, output = (p.resolve() for p in (args.checkpoint, args.vendor, args.output))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    assert checkpoint_manifest["trained_tokens"] == 10_000_000_000
    assert checkpoint_manifest["completed_steps"] == 305176
    assert checkpoint.name == "final_checkpoint"
    data = load_benchmarks(vendor)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model_config = json.loads((checkpoint / "config.json").read_text())
    prepared = {}
    prompt_stats = {}
    for benchmark, items in data.items():
        if args.limit is not None:
            items = items[:args.limit]
        prompts = [build_prompt(benchmark, item) for item in items]
        lengths = [len(tokens) for tokens in tokenizer(prompts, add_special_tokens=False)["input_ids"]]
        assert lengths and max(lengths) + args.max_new_tokens <= model_config["max_position_embeddings"]
        prepared[benchmark] = [{"dataset_index": i, "dataset_id": item.get("id", i),
                                "question": item.get("question", item.get("problem")),
                                "prompt": prompt, "prompt_tokens": length}
                               for i, (item, prompt, length) in enumerate(zip(items, prompts, lengths))]
        prompt_stats[benchmark] = {"questions": len(items), "minimum_tokens": min(lengths),
                                   "maximum_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths)}
        if not args.resume:
            with (output / f"{benchmark}_prompts.jsonl").open("w") as handle:
                for item in prepared[benchmark]:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    sources = {str(path): digest(path) for path in (
        vendor / "evaluation/datasets/gsm8k/test.jsonl", vendor / "evaluation/datasets/math/test.jsonl",
        vendor / "evaluation/few_shot_prompts/cot_gsm_8_shot.py",
        vendor / "evaluation/few_shot_prompts/cot_minerva_math_4_shot.py",
        vendor / "evaluation/few_shot_prompts/few_shot_prompting.py",
        Path(__file__), Path(__file__).with_name("math_benchmark_protocol.py"))}
    official_commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    assert official_commit == "b8b0f8ce093d80bf8e9a641e44142f06d092c305"
    protocol = {"checkpoint": str(checkpoint), "trained_input_tokens": 10_000_000_000,
                "checkpoint_manifest_sha256": digest(checkpoint / "checkpoint_manifest.json"),
                "model_shards_sha256": {p.name: digest(p) for p in sorted(checkpoint.glob("model-*.safetensors"))},
                "official_repository_commit": official_commit,
                "benchmarks": {key: len(value) for key, value in prepared.items()},
                "prompt_format": {"gsm8k": "Official DeepSeekMath Base 8-shot CoT",
                                  "math": "Official DeepSeekMath Base Minerva 4-shot CoT"},
                "chat_template": False, "do_sample": False, "num_beams": 1,
                "max_new_tokens": args.max_new_tokens, "n": 1, "batch_size": args.batch_size,
                "stops": {key: stop_strings(key) for key in prepared}, "seed": 42,
                "backend": "Unmodified native Hugging Face generate with SDPA and KV cache",
                "precision": "FP32 saved parameters retained; BF16 autocast compute, matching CPT precision",
                "torch": version("torch"), "transformers": version("transformers"),
                "smoke_limit_per_dataset": args.limit, "prompt_statistics": prompt_stats,
                "sources_sha256": sources,
                "limits": ["This checkpoint has CPT only, with no SFT/RL.",
                           "No GSM8K/MATH decontamination of the CPT corpus was performed.",
                           "HF backend differs from the paper's vLLM backend; outputs can differ with numerical rounding.",
                           "Answers exceeding 1024 new tokens are truncated and remain in the score denominator."]}
    manifest_path = output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        assert previous["protocol"] == protocol, "Resume protocol changed"
        manifest = previous
    else:
        assert not any((output / f"{key}_predictions.jsonl").exists() for key in prepared), "Existing predictions: use --resume or a new output folder"
        manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol}
        write_json(manifest_path, manifest)
    print(json.dumps({"event": "prepared", "prompt_statistics": prompt_stats, "prepare_only": args.prepare_only}), flush=True)
    if args.prepare_only:
        return
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                                torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval()
    model.config.use_cache = True
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    print(json.dumps({"event": "model_loaded", "seconds": time.perf_counter() - start,
                      "gpu": torch.cuda.get_device_name(), "parameter_dtype": "float32"}), flush=True)
    total_questions = sum(len(items) for items in prepared.values())
    total_finished = 0
    generated_tokens = 0
    generation_seconds = 0.0
    summaries = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for benchmark, items in prepared.items():
            prediction_path = output / f"{benchmark}_predictions.jsonl"
            existing = {}
            if args.resume and prediction_path.exists():
                for line in prediction_path.read_text().splitlines():
                    record = json.loads(line)
                    index = record["dataset_index"]
                    assert index not in existing and 0 <= index < len(items)
                    assert record["question"] == items[index]["question"] and record["dataset_id"] == items[index]["dataset_id"]
                    existing[index] = record
            total_finished += len(existing)
            remaining = sorted((item for item in items if item["dataset_index"] not in existing),
                               key=lambda item: (item["prompt_tokens"], item["dataset_index"]))
            generation_config = GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True,
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, stop_strings=stop_strings(benchmark))
            with prediction_path.open("a") as handle:
                for offset in range(0, len(remaining), args.batch_size):
                    batch = remaining[offset:offset + args.batch_size]
                    inputs = tokenizer([item["prompt"] for item in batch], padding=True,
                                       add_special_tokens=False, return_tensors="pt").to("cuda")
                    prompt_width = inputs["input_ids"].shape[1]
                    torch.cuda.synchronize()
                    batch_start = time.perf_counter()
                    sequences = model.generate(**inputs, generation_config=generation_config,
                                               tokenizer=tokenizer, logits_to_keep=1)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - batch_start
                    new_tokens = sequences[:, prompt_width:].cpu().tolist()
                    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    batch_tokens = 0
                    for item, token_ids, raw in zip(batch, new_tokens, completions):
                        eos_position = next((i for i, token in enumerate(token_ids) if token == tokenizer.eos_token_id), None)
                        token_count = len(token_ids) if eos_position is None else eos_position + 1
                        stop_positions = [raw.find(stop) for stop in stop_strings(benchmark) if stop in raw]
                        if stop_positions:
                            completion, finish = raw[:min(stop_positions)], "stop_string"
                            if eos_position is not None:
                                token_count -= 1  # EOS here is padding after the stop-string completion.
                        elif eos_position is not None:
                            completion, finish = raw, "eos"
                        else:
                            assert len(token_ids) == args.max_new_tokens
                            completion, finish = raw, "length"
                        record = {"benchmark": benchmark, "dataset_index": item["dataset_index"],
                                  "dataset_id": item["dataset_id"], "question": item["question"],
                                  "prompt_tokens": item["prompt_tokens"], "completion": completion,
                                  "raw_completion": raw, "finish_reason": finish, "generated_tokens": token_count,
                                  "generation_batch_seconds": seconds, "generation_batch_size": len(batch)}
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        existing[item["dataset_index"]] = record
                        batch_tokens += token_count
                    handle.flush()
                    os.fsync(handle.fileno())
                    total_finished += len(batch)
                    generated_tokens += batch_tokens
                    generation_seconds += seconds
                    progress = {"status": "generating", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                "benchmark": benchmark, "dataset_finished": len(existing), "dataset_total": len(items),
                                "finished": total_finished, "total": total_questions, "batch_seconds": seconds,
                                "generated_tokens_this_invocation": generated_tokens,
                                "generation_seconds_this_invocation": generation_seconds,
                                "generation_tokens_per_second": generated_tokens / generation_seconds,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
                    write_json(output / "progress.json", progress)
                    print(json.dumps({"event": "generation_progress", **progress}), flush=True)
                    del sequences, inputs
            assert sorted(existing) == list(range(len(items))), "Benchmark generation incomplete"
            summaries[benchmark] = {"questions": len(existing),
                                    "finish_reasons": dict(Counter(item["finish_reason"] for item in existing.values())),
                                    "generated_tokens": sum(item["generated_tokens"] for item in existing.values()),
                                    "prediction_sha256": digest(prediction_path)}
    result = {"status": "generation_complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "questions": total_questions, "benchmarks": summaries, "manifest": str(manifest_path),
              "generation_seconds_this_invocation": generation_seconds,
              "generation_tokens_per_second": generated_tokens / generation_seconds if generation_seconds else None,
              "wall_seconds_this_invocation": time.perf_counter() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "scoring": "pending", "training_modified": False}
    write_json(output / "generation_result.json", result)
    write_json(output / "progress.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
