"""Evaluate fixed validation strata through ordinary native HF, in one sequential pass."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOKEN_NAMES = ("pre_join_or_regular", "boundary_target", "post_join_context")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def token_markers(groups):
    """Index logits by their next-token targets; the last logit is unsupervised."""
    markers = np.zeros((groups["blocks"], groups["block_length"]), dtype=np.uint8)
    for block, positions in groups["missing_eos_join_target_positions"].items():
        positions = sorted(positions)
        markers[int(block), positions[0]:groups["block_length"] - 1] = 2
        markers[int(block), np.asarray(positions, dtype=np.int64) - 1] = 1
    return markers


def build_groups(data, diagnostics, output):
    meta = json.loads((data / "manifest.json").read_text())
    diag = json.loads(diagnostics.read_text())
    tokens = np.load(data / "validation_tokens.npy", mmap_mode="r")
    length, eos = meta["block_length"], diag["packing"]["eos_token_id"]
    assert length == 2048 and len(tokens) == 2097152
    assert sha(data / "validation_tokens.npy") == diag["data_files"]["validation_tokens_sha256"]
    joins = []
    with (data / "validation.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            end = row["token_offset"] + row["used_tokens"]
            if int(tokens[end - 1]) != eos and end < len(tokens) and end % length:
                joins.append(end)
    join_positions = {}
    for offset in joins:
        block, local = divmod(offset, length)
        join_positions.setdefault(str(block), []).append(local)
    missing = sorted(map(int, join_positions))
    assert missing == diag["packing"]["validation_truncated_documents"]["affected_block_indices"]
    pilot_blocks, blocks = 128, len(tokens) // length
    missing_set = set(missing)
    partition = []
    for subset in ("pilot", "extension"):
        population = range(pilot_blocks) if subset == "pilot" else range(pilot_blocks, blocks)
        for boundary in ("missing_eos", "regular"):
            indices = [i for i in population if (i in missing_set) == (boundary == "missing_eos")]
            partition.append({"name": f"{subset}_{boundary}", "subset": subset, "boundary": boundary,
                "block_indices": indices, "indices_int64_sha256": array_sha(np.asarray(indices, dtype=np.int64)),
                "blocks": len(indices), "input_tokens": len(indices) * length,
                "supervised_tokens": len(indices) * (length - 1)})
    flat = [i for group in partition for i in group["block_indices"]]
    assert len(flat) == len(set(flat)) == blocks and sorted(flat) == list(range(blocks))
    groups = {"schema_version": 1, "data": str(data.resolve()), "block_length": length,
        "blocks": blocks, "pilot_blocks": pilot_blocks, "input_tokens": len(tokens),
        "supervised_tokens": blocks * (length - 1), "eos_token_id": eos,
        "data_manifest_sha256": sha(data / "manifest.json"),
        "validation_tokens_sha256": sha(data / "validation_tokens.npy"),
        "validation_document_index_sha256": sha(data / "validation.jsonl"),
        "data_diagnostics_sha256": sha(diagnostics), "partition": partition,
        "missing_eos_join_offsets": joins, "missing_eos_join_target_positions": join_positions,
        "marginal_block_indices": {"pilot": list(range(pilot_blocks)), "extension": list(range(pilot_blocks, blocks)),
            "missing_eos": missing, "regular": [i for i in range(blocks) if i not in missing_set]},
        "token_group_definitions": {
            "boundary_target": "Next-document first target immediately after a quota-truncated prefix without EOS, inside a block.",
            "post_join_context": "All supervised targets after the first missing-EOS join in that block, excluding all boundary targets.",
            "pre_join_or_regular": "All remaining supervised targets: before the first bad join, or in unaffected blocks."},
        "cpu_checks": {"four_groups_nonoverlapping": True, "four_groups_cover_all_1024_blocks": True,
            "missing_join_indices_match_prior_diagnostics": True},
        "created_utc": datetime.now(timezone.utc).isoformat()}
    markers = token_markers(groups)[:, :length - 1]
    counts = np.bincount(markers.reshape(-1), minlength=3)
    assert int(counts.sum()) == groups["supervised_tokens"] and int(counts[1]) == len(joins)
    groups["token_group_counts"] = {name: int(counts[i]) for i, name in enumerate(TOKEN_NAMES)}
    groups["token_markers_uint8_sha256"] = array_sha(token_markers(groups))
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, groups)
    print(json.dumps({"groups": str(output), "partition_blocks": {g["name"]: g["blocks"] for g in partition},
                      "token_group_counts": groups["token_group_counts"]}, ensure_ascii=False))


def evaluate(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    groups = json.loads(args.groups.read_text())
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        assert sha(args.data / "manifest.json") == groups["data_manifest_sha256"]
        assert sha(args.data / "validation_tokens.npy") == groups["validation_tokens_sha256"]
        flat = [i for group in groups["partition"] for i in group["block_indices"]]
        assert len(flat) == len(set(flat)) == 1024 and sorted(flat) == list(range(1024))
        for group in groups["partition"]:
            assert array_sha(np.asarray(group["block_indices"], dtype=np.int64)) == group["indices_int64_sha256"]
        markers = token_markers(groups)
        assert array_sha(markers) == groups["token_markers_uint8_sha256"]
        sys.path.insert(0, str(args.source.resolve()))
        import cpt
        from packed_data import PackedTokens
        dataset = PackedTokens(args.data, split="validation", sequence_length=2048)
        dataset.explicit_mask = True
        assert len(dataset) == groups["blocks"] and args.micro_batch == 4
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native evaluation needs a CUDA BF16 device")
        if not Path(args.model).is_dir() and args.revision is None:
            raise ValueError("Pin --revision for a Hub base model")
        torch.set_num_threads(8)
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
        model.config.use_cache = False
        model.eval()
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(getattr(module, "_compiled_call_impl", None) is None for module in model.modules())
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        block_sums = np.zeros(groups["blocks"], dtype=np.float64)
        target_counts = np.zeros(groups["blocks"], dtype=np.int64)
        token_sums = np.zeros(3, dtype=np.float64)
        token_counts = np.zeros(3, dtype=np.int64)
        native_total = torch.zeros((), dtype=torch.float32, device="cuda")
        inputs = targets = 0
        eval_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(dataset), args.micro_batch):
                indices = range(start, min(start + args.micro_batch, len(dataset)))
                values, count, input_count = cpt.batch(dataset, indices)
                assert values["input_ids"].data_ptr() != values["labels"].data_ptr()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**values, use_cache=False)
                native_total += output.loss * count
                shifted = F.pad(values["labels"], (0, 1), value=-100)[..., 1:].contiguous()
                token_losses = F.cross_entropy(output.logits.float().reshape(-1, model.config.vocab_size),
                    shifted.reshape(-1), ignore_index=-100, reduction="none").reshape(len(indices), 2048).double()
                group_tensor = torch.as_tensor(markers[start:start + len(indices)], device="cuda")
                valid = shifted != -100
                by_block = token_losses.sum(dim=1)
                by_token_group = [token_losses[(group_tensor == i) & valid].sum() for i in range(3)]
                combined = torch.cat((by_block, torch.stack(by_token_group))).cpu().numpy()
                if not np.isfinite(combined).all():
                    raise RuntimeError(f"Nonfinite native token loss at block {start}")
                block_sums[start:start + len(indices)] = combined[:len(indices)]
                target_counts[start:start + len(indices)] = 2047
                token_sums += combined[len(indices):]
                token_counts += np.bincount(markers[start:start + len(indices), :2047].reshape(-1), minlength=3)
                inputs += input_count
                targets += count
                del output, token_losses, by_block, by_token_group, group_tensor, shifted, values
                if (start + len(indices)) % 256 == 0:
                    print(json.dumps({"event": "evaluation_progress", "blocks": start + len(indices)}), flush=True)
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - eval_started
        full_native_loss = float(native_total / targets)
        reaggregated = float(block_sums.sum() / targets)
        token_reaggregated = float(token_sums.sum() / token_counts.sum())
        if not (math.isfinite(full_native_loss) and abs(full_native_loss - args.expected_loss) <= args.loss_atol
                and abs(reaggregated - full_native_loss) <= args.loss_atol
                and abs(token_reaggregated - reaggregated) <= 1e-12):
            raise RuntimeError(f"Native loss parity failed: expected={args.expected_loss}, native={full_native_loss}, strata={reaggregated}, tokens={token_reaggregated}")
        assert inputs == groups["input_tokens"] and targets == groups["supervised_tokens"]
        assert {name: int(token_counts[i]) for i, name in enumerate(TOKEN_NAMES)} == groups["token_group_counts"]
        def group_result(name, indices):
            idx = np.asarray(indices, dtype=np.int64)
            total, n = float(block_sums[idx].sum()), int(target_counts[idx].sum())
            return {"name": name, "blocks": len(idx), "input_tokens": len(idx) * 2048,
                    "supervised_tokens": n, "target_fraction": n / targets, "loss_sum": total,
                    "loss": total / n, "perplexity": math.exp(total / n)}
        cross = [group_result(g["name"], g["block_indices"]) for g in groups["partition"]]
        marginals = [group_result(name, indices) for name, indices in groups["marginal_block_indices"].items()]
        token_results = [{"name": name, "supervised_tokens": int(token_counts[i]),
             "target_fraction": int(token_counts[i]) / targets, "loss_sum": float(token_sums[i]),
             "loss": float(token_sums[i] / token_counts[i]), "perplexity": math.exp(float(token_sums[i] / token_counts[i]))}
             for i, name in enumerate(TOKEN_NAMES)]
        result = {"status": "complete", "model": args.model, "revision": args.revision,
           "model_load_seconds": load_seconds, "evaluation_seconds": evaluation_seconds,
           "total_seconds": time.perf_counter() - started, "ordinary_native_full_loss": full_native_loss,
           "strata_reaggregated_full_loss": reaggregated, "token_groups_reaggregated_full_loss": token_reaggregated,
           "expected_native_full_loss": args.expected_loss, "loss_tolerance": args.loss_atol,
           "native_expected_difference": full_native_loss - args.expected_loss,
           "strata_native_difference": reaggregated - full_native_loss,
           "loss_parity_passed": True, "input_tokens": inputs, "supervised_tokens": targets,
           "cross_groups": cross, "marginal_groups": marginals, "token_groups": token_results,
           "groups_file": str(args.groups.resolve()), "groups_sha256": sha(args.groups),
           "source": str(args.source.resolve()), "source_sha256": {n: sha(args.source / n) for n in ("cpt.py", "packed_data.py")},
           "script_sha256": sha(Path(__file__)), "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
           "evaluation": "Unpatched native HF SDPA, FP32 weights, BF16 autocast, inference_mode, sequential ordinary 4x2048 batches, independent cloned labels, all-ones mask. Native HF.loss accumulated as in prior full validation; per-token native CE recomputed from the same logits for strata.",
           "limits": ["Strata partition a fixed selected validation set; they do not establish that missing EOS caused the long-run loss trend.",
             "Blocks containing joins are distinct from boundary target tokens. The 226 immediate boundary targets are only about 0.0108% of supervised tokens; post-join context targets are reported separately.",
             "Per-token recomputation is diagnostic overhead; these timings are not a training throughput benchmark."],
           "completed_utc": datetime.now(timezone.utc).isoformat()}
        write(args.output / "result.json", result)
        group_rows = [{"partition": kind, **row} for kind, rows in (("cross", cross), ("marginal", marginals), ("token", token_results)) for row in rows]
        fields = ("partition", "name", "blocks", "input_tokens", "supervised_tokens", "target_fraction", "loss_sum", "loss", "perplexity")
        with (args.output / "groups.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(group_rows)
        group_for_block = {i: g["name"] for g in groups["partition"] for i in g["block_indices"]}
        with (args.output / "blocks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("block", "group", "supervised_tokens", "loss_sum", "loss"));writer.writeheader()
            for i in range(groups["blocks"]):
                writer.writerow({"block": i, "group": group_for_block[i], "supervised_tokens": int(target_counts[i]),
                                 "loss_sum": float(block_sums[i]), "loss": float(block_sums[i] / target_counts[i])})
        print(json.dumps({"status": "complete", "native_loss": full_native_loss, "strata_loss": reaggregated,
                          "output": str(args.output)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        write(args.output / "result.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
            "expected_native_full_loss": args.expected_loss, "elapsed_seconds": time.perf_counter() - started})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-groups", action="store_true", help="CPU-only construction and exact partition verification")
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/openwebmath_5b_s42")
    p.add_argument("--groups", type=Path, default=ROOT / "runs/cpt_native_long_20261009/validation_strata_groups.json")
    p.add_argument("--data-diagnostics", type=Path, default=ROOT / "runs/cpt_native_long_20261009/data_diagnostics.json")
    p.add_argument("--source", type=Path, default=ROOT / "runs/cpt_10b_20261009_resumed/source")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--expected-loss", type=float)
    p.add_argument("--loss-atol", type=float, default=1e-6)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.build_groups:
        build_groups(args.data, args.data_diagnostics, args.groups)
    else:
        if args.model is None or args.expected_loss is None or args.output is None:
            p.error("Evaluation requires --model, --expected-loss and --output")
        evaluate(args)


if __name__ == "__main__":
    main()
