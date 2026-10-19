"""Compare two complete, protocol-matched math inference runs without rescoring.

Only saved prompts, generations, scored rows and metadata are read. No model is
loaded. Correctness is the recorded official score, with the comma-normalized
GSM8K score reported separately. All accuracy changes are percentage points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean


EXPECTED_COUNTS = {"gsm8k": 1319, "math": 5000}
MATCHED_PROTOCOL_FIELDS = (
    "official_repository_commit", "benchmarks", "prompt_format", "chat_template",
    "do_sample", "num_beams", "max_new_tokens", "n", "batch_size", "stops", "seed",
    "backend", "torch", "transformers", "smoke_limit_per_dataset", "prompt_statistics",
)
IDENTITY_FIELDS = ("dataset_index", "dataset_id", "question", "prompt_tokens")
PRECISION = {"parameters": "float32", "autocast": "bfloat16", "kv_cache": "native HF cache"}
KNOWN_PRECISION_TEXTS = {
    "FP32 saved parameters retained; BF16 autocast compute, matching CPT precision",
    "FP32 parameters; BF16 autocast compute, matching CPT evaluation precision",
}
LENGTH_REASONS = {"length", "max_tokens", "max_new_tokens"}
READ_FILES = ("manifest.json", "scores.json") + tuple(
    f"{benchmark}_{kind}.jsonl"
    for benchmark in EXPECTED_COUNTS for kind in ("prompts", "predictions", "scored")
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_indexed(path, expected):
    rows = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("dataset_index")
            require(type(index) is int and 0 <= index < expected,
                    f"{path}:{line_number}: invalid dataset_index {index!r}")
            require(index not in rows, f"{path}: duplicate dataset_index {index}")
            rows[index] = row
    require(set(rows) == set(range(expected)), f"{path}: expected all {expected} questions, got {len(rows)}")
    return [rows[index] for index in range(expected)]


def normalized_precision(protocol):
    require(protocol.get("precision") in KNOWN_PRECISION_TEXTS,
            f"Unrecognized precision description: {protocol.get('precision')!r}")
    value = protocol.get("compute_precision", PRECISION)
    require(value == PRECISION, f"Unexpected compute precision: {value!r}")
    return dict(value)


def official_source_hashes(protocol):
    # Absolute vendor locations differ between runs; these five relative sources
    # define the benchmark data and prompts. Generation-script hashes can differ
    # because the original checkpoint needs a different model-loading guard.
    result = {}
    for name, value in protocol["sources_sha256"].items():
        if "/evaluation/" in name:
            relative = "evaluation/" + name.split("/evaluation/", 1)[1]
            require(relative not in result, f"Duplicate official source hash: {relative}")
            result[relative] = value
    require(len(result) == 5, "Expected hashes of both datasets and all three prompt sources")
    return result


def validate_protocol(base, cpt):
    checked = {}
    for key in MATCHED_PROTOCOL_FIELDS:
        require(key in base and key in cpt, f"Missing protocol field {key}")
        require(base[key] == cpt[key], f"Protocol mismatch for {key}: {base[key]!r} versus {cpt[key]!r}")
        checked[key] = base[key]
    require(base["benchmarks"] == EXPECTED_COUNTS, "Both benchmarks must be the complete official test sets")
    require(base["smoke_limit_per_dataset"] is None, "Cannot compare smoke runs as full benchmarks")
    require(base["do_sample"] is False and base["num_beams"] == 1 and base["n"] == 1,
            "Expected one greedy completion per question")
    require(base["max_new_tokens"] == 1024 and base["batch_size"] in (256, 512),
            "Expected the 1024-token protocol with matched batch size 256 or 512")
    checked["compute_precision"] = normalized_precision(base)
    require(checked["compute_precision"] == normalized_precision(cpt), "Compute precision mismatch")
    checked["official_sources_sha256"] = official_source_hashes(base)
    require(checked["official_sources_sha256"] == official_source_hashes(cpt), "Benchmark/prompt source hashes differ")
    return checked


def accuracy(rows, correct_key="correct"):
    total = len(rows)
    correct = sum(row[correct_key] for row in rows)
    return {"total": total, "correct": correct, "accuracy": correct / total if total else None,
            "accuracy_percent": correct / total * 100 if total else None}


def rate(count, total):
    return 100 * count / total if total else None


def generation_summary(rows):
    total = len(rows)
    lengths = [row for row in rows if row["finish_reason"] in LENGTH_REASONS]
    nonlengths = [row for row in rows if row["finish_reason"] not in LENGTH_REASONS]
    finish_groups = defaultdict(list)
    for row in rows:
        finish_groups[row["finish_reason"]].append(row)
    return {
        "total": total, "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "mean_generated_tokens": mean(row["generated_tokens"] for row in rows) if rows else None,
        "mean_generated_tokens_correct": mean(row["generated_tokens"] for row in rows if row["correct"])
            if any(row["correct"] for row in rows) else None,
        "mean_generated_tokens_incorrect": mean(row["generated_tokens"] for row in rows if not row["correct"])
            if any(not row["correct"] for row in rows) else None,
        "length": {**accuracy(lengths), "percent_of_dataset": rate(len(lengths), total)},
        "nonlength": {**accuracy(nonlengths), "percent_of_dataset": rate(len(nonlengths), total)},
        "empty_predictions": sum(row["empty_prediction"] for row in rows),
        "empty_prediction_percent": rate(sum(row["empty_prediction"] for row in rows), total),
        "empty_completions": sum(not row["completion"].strip() for row in rows),
        "empty_completion_percent": rate(sum(not row["completion"].strip() for row in rows), total),
        "timeouts": sum(row["timeout"] for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "missing_generations": sum(row["missing_generation"] for row in rows),
        "finish_reasons": {reason: {**accuracy(group), "percent_of_dataset": rate(len(group), total),
            "mean_generated_tokens": mean(row["generated_tokens"] for row in group)}
            for reason, group in sorted(finish_groups.items())},
    }


def transition(base_correct, cpt_correct):
    if base_correct and cpt_correct:
        return "both_correct"
    if base_correct:
        return "base_only_correct"
    if cpt_correct:
        return "cpt_only_correct"
    return "both_wrong"


def paired_summary(base_rows, cpt_rows, key="correct"):
    require(len(base_rows) == len(cpt_rows), "Internal pair count mismatch")
    cells = Counter(transition(base[key], cpt[key]) for base, cpt in zip(base_rows, cpt_rows))
    result = {"total": len(base_rows), "base": accuracy(base_rows, key), "cpt": accuracy(cpt_rows, key),
              "paired": {name: cells[name] for name in
                  ("both_correct", "base_only_correct", "cpt_only_correct", "both_wrong")}}
    result["delta_correct"] = result["cpt"]["correct"] - result["base"]["correct"]
    result["delta_percentage_points"] = (
        result["cpt"]["accuracy_percent"] - result["base"]["accuracy_percent"] if base_rows else None)
    result["improved"] = cells["cpt_only_correct"]
    result["regressed"] = cells["base_only_correct"]
    result["unchanged"] = cells["both_correct"] + cells["both_wrong"]
    return result


def verify_score_summary(recorded, rows, label):
    expected = {**accuracy(rows), "timeouts": sum(row["timeout"] for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "empty_predictions": sum(row["empty_prediction"] for row in rows),
        "truncated_generations": sum(row["truncated"] for row in rows),
        "missing_generations": sum(row["missing_generation"] for row in rows)}
    for key, value in expected.items():
        require(key in recorded and recorded[key] == value,
                f"{label}: scores.json {key}={recorded.get(key)!r}, rows imply {value!r}")


def load_run(path):
    manifest = read_json(path / "manifest.json")
    scores = read_json(path / "scores.json")
    require(scores.get("status") == "complete", f"{path}: scoring is not complete")
    rows = {}
    for benchmark, expected in EXPECTED_COUNTS.items():
        prompt = read_indexed(path / f"{benchmark}_prompts.jsonl", expected)
        predictions = read_indexed(path / f"{benchmark}_predictions.jsonl", expected)
        scored = read_indexed(path / f"{benchmark}_scored.jsonl", expected)
        for index, (p, g, s) in enumerate(zip(prompt, predictions, scored)):
            prefix = f"{path}/{benchmark}:{index}"
            for field in IDENTITY_FIELDS:
                require(field in p and p[field] == g.get(field) == s.get(field),
                        f"{prefix}: prompt/prediction/score mismatch for {field}")
            require(isinstance(p.get("prompt"), str) and p["prompt"], f"{prefix}: missing prompt text")
            require(type(p["prompt_tokens"]) is int and p["prompt_tokens"] > 0, f"{prefix}: bad prompt length")
            require(g.get("benchmark") == s.get("benchmark") == benchmark, f"{prefix}: wrong benchmark")
            for field, value in g.items():
                require(field in s and s[field] == value, f"{prefix}: scored generation changed {field}")
            require(isinstance(s.get("completion"), str), f"{prefix}: nontext completion")
            require(s.get("finish_reason") in {"stop_string", "eos"} | LENGTH_REASONS,
                    f"{prefix}: unrecognized finish reason")
            require(type(s.get("generated_tokens")) is int and 0 <= s["generated_tokens"] <= 1024,
                    f"{prefix}: invalid generated token count")
            require(type(s.get("generation_batch_size")) is int and 1 <= s["generation_batch_size"] <= 512,
                    f"{prefix}: invalid actual batch size")
            for key in ("correct", "empty_prediction", "timeout", "missing_generation", "truncated"):
                require(type(s.get(key)) is bool, f"{prefix}: {key} is not bool")
            require(s["missing_generation"] is False, f"{prefix}: missing generation")
            require(s["truncated"] == (s["finish_reason"] in LENGTH_REASONS), f"{prefix}: truncation flag mismatch")
            require("gold" in s and "prediction" in s and "error" in s, f"{prefix}: incomplete scored row")
            if benchmark == "gsm8k":
                require(type(s.get("gsm8k_comma_normalized_correct")) is bool,
                        f"{prefix}: missing secondary correctness")
        recorded = scores["benchmarks"][benchmark]
        verify_score_summary(recorded, scored, f"{path}/{benchmark}")
        require(recorded.get("prediction_sha256") == digest(path / f"{benchmark}_predictions.jsonl"),
                f"{path}/{benchmark}: prediction file hash mismatch")
        if benchmark == "gsm8k":
            secondary = accuracy(scored, "gsm8k_comma_normalized_correct")
            for key in ("total", "correct", "accuracy"):
                require(recorded["secondary_comma_normalized"][key] == secondary[key],
                        f"{path}/{benchmark}: secondary score mismatch for {key}")
            changed = sum(s["prediction"] != s["gsm8k_comma_normalized_prediction"] for s in scored)
            require(recorded["secondary_comma_normalized"]["changed_predictions"] == changed,
                    f"{path}/{benchmark}: changed-prediction count mismatch")
        else:
            for field in ("type", "level", "category"):
                grouped = defaultdict(list)
                for s in scored:
                    require(isinstance(s.get(field), str) and s[field], f"{path}: missing MATH {field}")
                    grouped[s[field]].append(s)
                require(set(recorded[f"by_{field}"]) == set(grouped), f"{path}: {field} group keys mismatch")
                for group, values in grouped.items():
                    verify_score_summary(recorded[f"by_{field}"][group], values, f"{path}/{field}/{group}")
        stats = manifest["protocol"]["prompt_statistics"][benchmark]
        observed = {"questions": len(prompt), "minimum_tokens": min(p["prompt_tokens"] for p in prompt),
            "maximum_tokens": max(p["prompt_tokens"] for p in prompt),
            "mean_tokens": mean(p["prompt_tokens"] for p in prompt)}
        require(stats == observed, f"{path}/{benchmark}: prompt statistics mismatch")
        rows[benchmark] = {"prompts": prompt, "scored": scored}
    return {"manifest": manifest, "scores": scores, "benchmarks": rows}


def compare_runs(base, cpt):
    protocol = validate_protocol(base["manifest"]["protocol"], cpt["manifest"]["protocol"])
    for field in ("protocol", "per_item_timeout_seconds", "denominator", "generated_python_execution"):
        require(base["scores"].get(field) == cpt["scores"].get(field), f"Scoring protocol mismatch for {field}")
    results = {}
    pairs = {}
    breakdown = []
    for benchmark, expected in EXPECTED_COUNTS.items():
        bdata, cdata = (run["benchmarks"][benchmark] for run in (base, cpt))
        brows, crows = bdata["scored"], cdata["scored"]
        records = []
        for index, (bp, cp, b, c) in enumerate(zip(bdata["prompts"], cdata["prompts"], brows, crows)):
            for key in (*IDENTITY_FIELDS, "prompt"):
                require(bp[key] == cp[key], f"{benchmark}:{index}: cross-model {key} mismatch")
            require(b["gold"] == c["gold"], f"{benchmark}:{index}: gold mismatch")
            require(b["generation_batch_size"] == c["generation_batch_size"],
                    f"{benchmark}:{index}: actual generation batch sizes differ")
            state = transition(b["correct"], c["correct"])
            item = {"benchmark": benchmark, **{key: b[key] for key in IDENTITY_FIELDS},
                "gold": b["gold"], "prompt_sha256": hashlib.sha256(bp["prompt"].encode()).hexdigest(),
                "transition": state, "change": {"cpt_only_correct": "improved", "base_only_correct": "regressed"}.get(state, "unchanged"),
                "base": b, "cpt": c}
            if benchmark == "gsm8k":
                item["comma_normalized_transition"] = transition(
                    b["gsm8k_comma_normalized_correct"], c["gsm8k_comma_normalized_correct"])
            else:
                for key in ("type", "level", "category"):
                    require(b[key] == c[key], f"{benchmark}:{index}: {key} mismatch")
                    item[key] = b[key]
            records.append(item)
        paired = paired_summary(brows, crows)
        shared_indices = [index for index, (b, c) in enumerate(zip(brows, crows))
                          if b["finish_reason"] not in LENGTH_REASONS and c["finish_reason"] not in LENGTH_REASONS]
        paired["both_nonlength_same_questions"] = {
            **paired_summary([brows[i] for i in shared_indices], [crows[i] for i in shared_indices]),
            "percent_of_full_dataset": rate(len(shared_indices), expected),
            "interpretation": "Descriptive subset selected by both model outputs; not a causal estimate of removing the length limit.",
        }
        paired["generation"] = {"base": generation_summary(brows), "cpt": generation_summary(crows)}
        paired["length_status_paired"] = dict(Counter(
            f"base_{'length' if b['finish_reason'] in LENGTH_REASONS else 'nonlength'}__cpt_{'length' if c['finish_reason'] in LENGTH_REASONS else 'nonlength'}"
            for b, c in zip(brows, crows)))
        if benchmark == "gsm8k":
            paired["secondary_comma_normalized"] = paired_summary(brows, crows, "gsm8k_comma_normalized_correct")
            paired["secondary_comma_normalized"]["label"] = "Diagnostic only; not official main score"
            paired["secondary_comma_normalized"]["both_nonlength_same_questions"] = paired_summary(
                [brows[i] for i in shared_indices], [crows[i] for i in shared_indices], "gsm8k_comma_normalized_correct")
        else:
            groups = {}
            for field in ("type", "level"):
                groups[field] = {}
                for group in sorted({row[field] for row in brows}):
                    indices = [index for index, row in enumerate(brows) if row[field] == group]
                    summary = paired_summary([brows[i] for i in indices], [crows[i] for i in indices])
                    groups[field][group] = summary
                    breakdown.append(comparison_csv_row("math", summary, dimension=field, group=group))
            paired["breakdown"] = groups
        results[benchmark] = paired
        pairs[benchmark] = records
    return protocol, results, pairs, breakdown


def comparison_csv_row(benchmark, summary, score="official", **extra):
    return {"benchmark": benchmark, "score": score, **extra, "total": summary["total"],
        "base_correct": summary["base"]["correct"], "cpt_correct": summary["cpt"]["correct"],
        "base_accuracy_percent": summary["base"]["accuracy_percent"],
        "cpt_accuracy_percent": summary["cpt"]["accuracy_percent"],
        "delta_correct": summary["delta_correct"], "delta_percentage_points": summary["delta_percentage_points"],
        **summary["paired"], "improved": summary["improved"], "regressed": summary["regressed"],
        "unchanged": summary["unchanged"]}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_pairs(path, records):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path.with_suffix(".jsonl"))
    flat = []
    for record in records:
        row = {key: record.get(key, "") for key in (
            "benchmark", "dataset_index", "dataset_id", "question", "prompt_tokens", "prompt_sha256",
            "transition", "change", "type", "level", "category", "comma_normalized_transition")}
        row["gold"] = json.dumps(record["gold"], ensure_ascii=False)
        for name in ("base", "cpt"):
            for key in ("completion", "raw_completion", "prediction", "official_evaluated_prediction", "correct",
                        "finish_reason", "generated_tokens", "empty_prediction", "timeout", "error",
                        "gsm8k_comma_normalized_prediction", "gsm8k_comma_normalized_correct"):
                value = record[name].get(key, "")
                row[f"{name}_{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        flat.append(row)
    write_csv(path.with_suffix(".csv"), flat)


def model_metadata(run):
    protocol = run["manifest"]["protocol"]
    fields = ("checkpoint", "trained_input_tokens", "checkpoint_manifest_sha256", "model_shards_sha256",
              "checkpoint_stored_dtype", "precision", "compute_precision")
    result = {key: protocol.get(key) for key in fields}
    if result["checkpoint_stored_dtype"] is None and protocol["precision"] == (
            "FP32 saved parameters retained; BF16 autocast compute, matching CPT precision"):
        result["checkpoint_stored_dtype"] = "float32"
        result["checkpoint_stored_dtype_source"] = "Legacy CPT manifest precision description"
    result["normalized_compute_precision"] = normalized_precision(protocol)
    return result


def plot_comparison(output, results, self_check):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    colors = ("#64748b", "#2563eb")
    for row, benchmark in enumerate(("gsm8k", "math")):
        stats = results[benchmark]
        for model, offset, color, label in (
            ("base", -.17, colors[0], "Original base model"),
            ("cpt", .17, colors[1], "After 10B-token math CPT"),
        ):
            value = stats[model]["accuracy_percent"]
            ax.barh(row + offset, value, height=.29, color=color, label=label if row == 0 else None)
            ax.text(value + .65, row + offset, f"{value:.2f}%  ({stats[model]['correct']:,}/{stats['total']:,})",
                    va="center", fontsize=10)
        ax.text(1.01, row, f"{stats['delta_percentage_points']:+.2f} pp", va="center", ha="right",
                transform=ax.get_yaxis_transform(), color="#1e40af", fontweight="bold")
    ax.set_yticks([0, 1], ["GSM8K · 8-shot", "MATH · 4-shot"])
    ax.invert_yaxis()
    largest = max(results[b][m]["accuracy_percent"] for b in results for m in ("base", "cpt"))
    ax.set_xlim(0, min(100, max(60, largest + 23)))
    ax.set_xlabel("Official accuracy (%) · all test questions")
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False, fontsize=10)
    title = "Original model vs. final 10B checkpoint"
    if self_check:
        title = "SELF-CHECK ONLY · identical inference run compared with itself"
    fig.suptitle(title, y=.97, fontweight="bold", fontsize=13)
    auxiliary = results["gsm8k"]["secondary_comma_normalized"]
    fig.text(.02, .085, "Greedy decoding · native HF / SDPA · FP32 parameters + BF16 autocast · max 1,024 new tokens", fontsize=9)
    fig.text(.02, .035,
        f"GSM8K comma-normalized diagnostic: {auxiliary['base']['accuracy_percent']:.2f}% → "
        f"{auxiliary['cpt']['accuracy_percent']:.2f}% ({auxiliary['delta_percentage_points']:+.2f} pp); separate from official score.",
        fontsize=9, color="#475569")
    fig.subplots_adjust(left=.19, right=.98, bottom=.25, top=.74)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"benchmark_comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cpt-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true", help="Allow identical input runs; label all output as a self-check")
    args = parser.parse_args()
    base_path, cpt_path, output = (path.resolve() for path in (args.base_run, args.cpt_run, args.output))
    require(base_path != cpt_path or args.self_check, "Identical runs require --self-check; this is not a model comparison")
    require(not args.self_check or base_path == cpt_path, "--self-check is only for an identical-run zero-delta check")
    for path in (base_path, cpt_path):
        require(output != path and path not in output.parents, "Report output must be outside both input run directories")
    paths = [path / filename for path in dict.fromkeys((base_path, cpt_path)) for filename in READ_FILES]
    before = {str(path): digest(path) for path in paths}
    base, cpt = load_run(base_path), load_run(cpt_path)
    if not args.self_check:
        bprotocol, cprotocol = (run["manifest"]["protocol"] for run in (base, cpt))
        require(bprotocol.get("trained_input_tokens") == 0, "Original-model run must declare zero additional CPT input tokens")
        require(cprotocol.get("trained_input_tokens") == 10_000_000_000, "CPT run must use the final 10B checkpoint")
        require(bprotocol.get("model_shards_sha256") and cprotocol.get("model_shards_sha256"),
                "Both model identities must include saved-weight hashes")
        require(set(bprotocol["model_shards_sha256"].values()) != set(cprotocol["model_shards_sha256"].values()),
                "Original and CPT runs declare identical saved-weight hashes")
    protocol, results, pairs, breakdown = compare_runs(base, cpt)
    after = {str(path): digest(path) for path in paths}
    require(before == after, "Input files changed during comparison; retry after inference/scoring finish")
    if args.self_check:
        for benchmark, result in results.items():
            require(result["delta_correct"] == 0 and result["improved"] == result["regressed"] == 0,
                    f"{benchmark}: self-check failed")
    output.mkdir(parents=True, exist_ok=True)
    for benchmark, records in pairs.items():
        write_pairs(output / f"{benchmark}_paired", records)
    table = [comparison_csv_row(benchmark, result) for benchmark, result in results.items()]
    table.append(comparison_csv_row("gsm8k", results["gsm8k"]["secondary_comma_normalized"], "comma_normalized_diagnostic"))
    write_csv(output / "benchmark_comparison.csv", table)
    write_csv(output / "math_breakdown_comparison.csv", breakdown)
    plot_comparison(output, results, args.self_check)
    report = {
        "status": "self_check_complete" if args.self_check else "complete_protocol_matched_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_run": str(base_path), "cpt_run": str(cpt_path), "questions_per_model": sum(EXPECTED_COUNTS.values()),
        "comparison_direction": "CPT minus original; changes in accuracy are percentage points",
        "self_check": args.self_check, "matched_protocol": protocol,
        "input_model_metadata": {"base": model_metadata(base), "cpt": model_metadata(cpt)},
        "validation": {"complete_unique_index_sets": EXPECTED_COUNTS,
            "prompt_identity_and_text_and_token_lengths_equal": True, "gold_and_math_groups_equal": True,
            "generation_records_preserved_by_scoring": True, "recorded_scores_recomputed_from_saved_booleans": True,
            "prediction_hashes_match_scores": True, "input_files_unchanged": True,
            "scoring_protocol_equal": True, "actual_generation_batch_sizes_equal": True,
            "model_token_budgets_and_distinct_saved_weights_checked": not args.self_check,
            "model_loaded": False, "scoring_rerun": False},
        "benchmarks": results, "input_files_sha256": before,
        "script_sha256": digest(Path(__file__)),
        "limits": [
            "This comparison describes these two checkpoints under one shared prompt/decoding/scoring protocol.",
            "All questions, including length-limited and empty answers, remain in the primary score denominator.",
            "The both-nonlength subset is selected using model outputs and is descriptive, not a causal length-limit ablation.",
            "GSM8K comma normalization changes the answer extraction; it is a separate diagnostic score.",
            "One CPT run cannot identify which training choice caused a change or establish generalization across runs.",
            "No evaluation-set decontamination of the training corpus was performed.",
        ],
        "artifacts": {name: str(output / name) for name in (
            "benchmark_comparison.csv", "math_breakdown_comparison.csv", "gsm8k_paired.jsonl", "gsm8k_paired.csv",
            "math_paired.jsonl", "math_paired.csv", "benchmark_comparison.png", "benchmark_comparison.pdf")},
    }
    write_json(output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "output": str(output),
        "benchmarks": {benchmark: {key: value[key] for key in
            ("total", "base", "cpt", "delta_percentage_points", "paired")}
            for benchmark, value in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
