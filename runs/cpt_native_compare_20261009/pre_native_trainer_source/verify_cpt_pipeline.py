"""Verify saved CPT stop/resume evidence using CPU metadata only.

Checkpoint tensors are mmap-loaded on CPU; optimizer/model tensor values are
not compared here.  This diagnostic verifies schedule, order, counts, logs and
native held-out loss continuity, not a bitwise match to an uninterrupted run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

import torch

from cpt_orders import EpochOrders, load_saved_orders, order_indices, prefix_counts
from source_migration import verify_lineage_snapshots, without_source


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_numbers(item) for item in value]
    return value


def saved_path(value):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    # Saved CLI arguments are normally relative to the shared workspace, even
    # when this verifier is invoked from another working directory.
    return (Path(__file__).resolve().parents[2] / path).resolve()


def read_run(directory):
    result = json.loads((directory / "result.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines() if line.strip()]
    checkpoint = saved_path(result["checkpoint"])
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    return {"directory": directory.resolve(), "result": result, "manifest": manifest,
            "rows": rows, "state": state, "checkpoint": checkpoint, "orders": load_saved_orders(state, checkpoint),
            "state_path": state_path,
            "validation_before": json.loads((directory / "validation_before.json").read_text())}


def planned_lr(schedule, one_based_step):
    if one_based_step <= schedule["warmup_steps"]:
        factor = one_based_step / schedule["warmup_steps"]
    elif one_based_step <= schedule["decay_80_step"]:
        factor = 1.0
    elif one_based_step <= schedule["decay_90_step"]:
        factor = schedule["decay_80_factor"]
    else:
        factor = schedule["decay_90_factor"]
    return schedule["peak_lr"] * factor


def metadata_timing(run):
    result = run["result"]
    return {"result_path": str(run["directory"] / "result.json"),
            "status": result.get("status"), "completed_steps": result.get("completed_steps"),
            "trained_tokens": result.get("trained_tokens"),
            "peak_allocated_gib": result.get("peak_allocated_bytes", 0) / (1024 ** 3),
            "peak_reserved_gib": result.get("peak_reserved_bytes", 0) / (1024 ** 3),
            "train_seconds_excluding_validation_and_save": result.get("train_seconds_excluding_validation_and_save"),
            "steady_tokens_per_second": result.get("steady_tokens_per_second"),
            "steady_tokens": result.get("steady_tokens"), "steady_seconds": result.get("steady_seconds"),
            "wall_seconds_this_invocation": result.get("wall_seconds_this_invocation"),
            "invocations": result.get("invocations"),
            "initial_update_rows": [row for row in run["rows"] if row.get("first_full_shape_update")],
            "optimizer_compile": result.get("optimizer_compile")}


def verify(args):
    first = read_run(args.first_run)
    resumed = read_run(args.resumed_run)
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    first_result, final_result = first["result"], resumed["result"]
    first_state, final_state = first["state"], resumed["state"]
    signature = first_state["signature"]
    schedule = first_state["schedule"]
    stopped_step = first_state["completed_steps"]
    expected_steps = schedule["total_steps"]
    dataset_tokens = first_result["dataset_input_tokens"]
    sequence_length = signature["sequence_length"]
    effective_batch = signature["effective_batch"]
    epochs = signature.get("epochs", 1)
    blocks = math.ceil(dataset_tokens / sequence_length)
    epoch_steps = math.ceil(blocks / effective_batch)
    order = first["orders"]
    expected_inputs, expected_targets, expected_updates = prefix_counts(order, expected_steps, effective_batch,
                                                                       sequence_length, dataset_tokens)
    check("stopped_then_complete_status", first_result.get("status") == "stopped" and final_result.get("status") == "complete")
    check("valid_stop_boundary", isinstance(stopped_step, int) and not isinstance(stopped_step, bool)
          and 0 < stopped_step < expected_steps, stopped_step=stopped_step)
    signatures = [first["manifest"]["signature"], resumed["manifest"]["signature"], final_state["signature"]]
    first_lineage = verify_lineage_snapshots(first_state, sha256)
    final_lineage = verify_lineage_snapshots(final_state, sha256)
    audits_match = all(
        run["manifest"].get("original_signature", run["state"]["signature"]) == audit["original_signature"]
        and run["manifest"].get("source_migrations", []) == audit["source_migrations"]
        for run, audit in ((first, first_lineage), (resumed, final_lineage)))
    first_chain, final_chain = first_lineage["source_migrations"], final_lineage["source_migrations"]
    lineage_extends_first = (first_lineage["original_signature"] == final_lineage["original_signature"]
                            and final_chain[:len(first_chain)] == first_chain
                            and all(entry["completed_steps"] >= stopped_step for entry in final_chain[len(first_chain):]))
    check("identical_or_audited_source_only_signatures", first["manifest"]["signature"] == signature
          and resumed["manifest"]["signature"] == final_state["signature"]
          and all(without_source(value) == without_source(signature) for value in signatures)
          and audits_match and lineage_extends_first,
          source_only_migration_count=final_lineage["source_only_migration_count"])
    schedules = [first_result["schedule"], final_result["schedule"], first["manifest"]["schedule"],
                 resumed["manifest"]["schedule"], final_state["schedule"]]
    check("identical_saved_schedules", all(value == schedule for value in schedules), schedule=schedule)
    check("unchanged_shuffle_order", first_state.get("order_metadata", first_state.get("order"))
          == final_state.get("order_metadata", final_state.get("order")))
    config_hash = sha256(args.backend_config)
    check("current_backend_config_matches_saved_signature", signature["backend_config_sha256"] == config_hash,
          saved_sha256=signature["backend_config_sha256"], current_sha256=config_hash,
          backend_config=str(args.backend_config.resolve()))
    checkpoint_snapshot_hashes = [sha256(run["directory"] / "backend_config.json") for run in (first, resumed)]
    check("saved_backend_config_snapshots_match", all(value == config_hash for value in checkpoint_snapshot_hashes))
    check("resumed_argument_points_to_stopped_checkpoint",
          saved_path(resumed["manifest"]["arguments"]["resume"]) == first["checkpoint"],
          resumed_argument=resumed["manifest"]["arguments"]["resume"], stopped_checkpoint=str(first["checkpoint"]))
    check("resume_started_at_saved_boundary", resumed["manifest"].get("resumed_completed_steps") == stopped_step
          and resumed["validation_before"].get("step") == stopped_step)
    stop_validation = first_result["validation_after"]
    resume_validation = resumed["validation_before"]
    check("exact_native_validation_continuity", stop_validation == resume_validation,
          stopped_native_validation=stop_validation, resumed_initial_native_validation=resume_validation)
    check("native_validation_backend", stop_validation.get("loss_backend") == "native eager"
          and resume_validation.get("loss_backend") == "native eager")
    check("configured_global_token_budget", 1 <= expected_steps <= epoch_steps * epochs
          and first_result["configured_input_tokens"] == final_result["configured_input_tokens"] == expected_inputs,
          epochs=epochs, expected_input_tokens=expected_inputs, expected_supervised_tokens=expected_targets)
    check("shuffle_order_is_exact_dataset_permutation", (order.blocks == blocks and order.epochs == epochs
          if isinstance(order, EpochOrders) else len(order) == blocks))
    check("schedule_covers_requested_epochs_or_explicit_cap", schedule["total_steps"] == min(
          first["manifest"]["arguments"].get("steps") or epoch_steps * epochs, epoch_steps * epochs))
    first_rows, resumed_rows = first["rows"], resumed["rows"]
    rows = first_rows + resumed_rows
    check("first_log_covers_stopped_prefix", [row.get("step") for row in first_rows] == list(range(1, stopped_step + 1)))
    check("resumed_log_covers_remaining_suffix", [row.get("step") for row in resumed_rows]
          == list(range(stopped_step + 1, expected_steps + 1)))
    check("combined_logs_contiguous_global_steps", [row.get("step") for row in rows] == list(range(1, expected_steps + 1)))
    lr_errors = []
    token_errors = []
    cumulative_inputs = 0
    cumulative_targets = 0
    applied_updates = 0
    tail_rows = []
    remainder = dataset_tokens % sequence_length
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int) or not 1 <= step <= schedule["total_steps"]:
            lr_errors.append({"step": step, "error": "outside saved schedule"})
            continue
        rate = planned_lr(schedule, step)
        logged_rate = row.get("lr")
        if not isinstance(logged_rate, (int, float)) or not math.isfinite(logged_rate) or not math.isclose(logged_rate, rate, rel_tol=1e-12, abs_tol=1e-18):
            lr_errors.append({"step": step, "expected": rate, "actual": logged_rate})
        indices = order_indices(order, step - 1, effective_batch)
        row_inputs = len(indices) * sequence_length
        padded_tail = bool(remainder and blocks - 1 in indices)
        if padded_tail:
            row_inputs -= sequence_length - remainder
        row_targets = row_inputs - len(indices)
        cumulative_inputs += row_inputs
        cumulative_targets += row_targets
        applied_updates += int(row_targets > 0)
        if (row.get("input_tokens") != row_inputs or row.get("supervised_tokens") != row_targets
                or row.get("cumulative_tokens") != cumulative_inputs
                or row.get("optimizer_applied") is not (row_targets > 0)):
            token_errors.append({"step": step, "expected_input_tokens": row_inputs,
                                 "expected_supervised_tokens": row_targets,
                                 "expected_cumulative_tokens": cumulative_inputs, "row": row})
        if padded_tail or len(indices) < effective_batch:
            tail_rows.append({"kind": "mixed_padded_tail" if padded_tail and len(indices) == effective_batch
                              else "partial_final_batch", "block_indices": [int(x) for x in indices],
                              "padded_tail_present": padded_tail, "row": row})
    check("every_logged_lr_matches_saved_plan", not lr_errors, mismatches=lr_errors)
    check("every_logged_token_count_and_update_flag_matches_order", not token_errors, mismatches=token_errors)
    check("exact_configured_inputs_and_targets", cumulative_inputs == expected_inputs
          and cumulative_targets == expected_targets,
          input_tokens=cumulative_inputs, supervised_tokens=cumulative_targets)
    for label, run in (("stopped", first), ("final", resumed)):
        result, state = run["result"], run["state"]
        prefix = rows[:state["completed_steps"]]
        inputs = sum(row["input_tokens"] for row in prefix)
        targets = sum(row["supervised_tokens"] for row in prefix)
        applied = sum(int(row["optimizer_applied"]) for row in prefix)
        check(f"{label}_checkpoint_and_result_counters_match_logs",
              result["completed_steps"] == state["completed_steps"] == len(prefix)
              and result["trained_tokens"] == state["trained_tokens"] == inputs
              and result["supervised_tokens"] == state["supervised_tokens"] == targets
              and result["optimizer_updates"] == state["optimizer_updates"] == applied,
              completed_steps=state["completed_steps"], input_tokens=inputs,
              supervised_tokens=targets, optimizer_updates=applied)
    full_epochs = expected_steps == epoch_steps * epochs
    check("final_exact_budget_and_updates", final_state["completed_steps"] == expected_steps
          and final_state["trained_tokens"] == expected_inputs
          and final_state["supervised_tokens"] == expected_targets
          and final_state["optimizer_updates"] == applied_updates == expected_updates
          and final_result.get("all_dataset_tokens_consumed") is full_epochs,
          expected_optimizer_updates=expected_updates)
    finite_rows = all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                      for row in rows for key in ("loss", "grad_norm", "seconds"))
    check("logged_loss_gradient_norm_and_timing_finite", finite_rows)
    regular_seconds = [row["seconds"] for row in rows if not row.get("first_full_shape_update")
                       and row.get("input_tokens") == sequence_length * effective_batch
                       and isinstance(row.get("seconds"), (int, float)) and math.isfinite(row["seconds"])]
    regular_median = statistics.median(regular_seconds) if regular_seconds else None
    for item in tail_rows:
        duration = item["row"].get("seconds")
        item["seconds_vs_regular_step_median"] = duration / regular_median if regular_median and isinstance(duration, (int, float)) else None
    checks_passed = all(item["passed"] for item in checks)
    return strict_numbers({
        "schema_version": 1, "status": "passed" if checks_passed else "failed", "passed": checks_passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "first_run": str(args.first_run.resolve()), "resumed_run": str(args.resumed_run.resolve()),
        "backend_config": str(args.backend_config.resolve()), "backend_config_sha256": config_hash,
        "checks": checks, "check_count": len(checks),
        "checkpoint_loading": "torch.load(map_location='cpu', mmap=True, weights_only=True); metadata/order/counters only, no full optimizer tensor comparison",
        "timings": {"first_run": metadata_timing(first), "resumed_run": metadata_timing(resumed),
                    "regular_step_median_seconds": regular_median,
                    "mixed_padded_tail_and_final_partial_rows": tail_rows,
                    "timing_threshold_applied": False,
                    "note": "Tail durations are reported as evidence; no speed threshold changes verification status"},
        "evidence": {label: {"checkpoint_state_path": str(run["state_path"]),
                              "manifest_path": str(run["directory"] / "manifest.json"),
                              "manifest_sha256": sha256(run["directory"] / "manifest.json"),
                              "training_log_path": str(run["directory"] / "training.jsonl"),
                              "training_log_sha256": sha256(run["directory"] / "training.jsonl")}
                     for label, run in (("first", first), ("resumed", resumed))},
        "limitations": ["This does not prove bitwise training-trajectory equality to a separate uninterrupted run",
                        "Model and optimizer tensor equality belongs to the separate checkpoint diagnostic",
                        "Reported wall time covers each saved invocation; combined training counters include the earlier stopped prefix"],
        "gpu_execution": False,
    })


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-run", type=Path, required=True)
    p.add_argument("--resumed-run", type=Path, required=True)
    p.add_argument("--backend-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        result = verify(args)
    except Exception as error:
        result = {"status": "failed", "passed": False, "error": str(error),
                  "traceback": traceback.format_exc(), "gpu_execution": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(strict_numbers(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                      "check_count": result.get("check_count")}, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
