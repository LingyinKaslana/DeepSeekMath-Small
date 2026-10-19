"""Aggregate saved CPT optimization evidence without loading PyTorch or CUDA.

Run again after additional evidence is ready.  All candidate results, including
numerical rejections and execution failures, are retained.  Nonfinite JSON
numbers are represented by null, with their original field paths recorded.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def strict_numbers(value, nonfinite_paths=None, path="$"):
    """Replace NaN/Infinity recursively while preserving all other evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite_paths is not None:
            nonfinite_paths.append(path.rstrip())
        return None
    if isinstance(value, dict):
        return {key: strict_numbers(item, nonfinite_paths, f"{path.rstrip()}.{key}")
                for key, item in value.items()}
    if isinstance(value, list):
        return [strict_numbers(item, nonfinite_paths, f"{path.rstrip()}[{index}]")
                for index, item in enumerate(value)]
    return value


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path, include_data=False):
    path = Path(path).resolve()
    evidence = {"path": str(path), "exists": path.is_file()}
    if evidence["exists"]:
        evidence["sha256"] = sha256(path)
        if include_data:
            fields = []
            evidence["data"] = strict_numbers(json.loads(path.read_text()), fields)
            evidence["nonfinite_source_fields"] = fields
    return evidence


def read_optional(path):
    return json.loads(path.read_text()) if path.is_file() else None


def summarize_candidate(path):
    result = json.loads(path.read_text())
    directory = path.parent
    manifest_path = directory / "manifest.json"
    manifest = read_optional(manifest_path)
    config = manifest.get("config", {}) if manifest else {}
    numerical = result.get("numerical")
    if numerical is None:
        numerical = read_optional(directory / "numerical_check.json")
    optimizer_check = result.get("optimizer_check")
    if optimizer_check is None:
        optimizer_check = read_optional(directory / "optimizer_check.json")
    allocated = result.get("peak_allocated_bytes")
    reserved = result.get("peak_reserved_bytes")
    optimizer_metadata = result.get("optimizer_metadata") or {}
    candidate = {
        "name": directory.name,
        "status": result.get("status", "unknown"),
        "tokens_per_second": result.get("tokens_per_second"),
        "seconds_per_update": result.get("seconds_per_update"),
        "measured_updates": result.get("measured_updates"),
        "measured_seconds": result.get("seconds"),
        "peak_allocated_gib": allocated / (1024 ** 3) if finite_number(allocated) else None,
        "peak_reserved_gib": reserved / (1024 ** 3) if finite_number(reserved) else None,
        "micro_batch": result.get("micro_batch", config.get("micro_batch")),
        "effective_batch": result.get("effective_batch", config.get("effective_batch")),
        "mask_policy": result.get("mask_policy", config.get("mask_policy")),
        "fused_optimizer": result.get("fused_optimizer", config.get("fused_optimizer", False)),
        "compiled_optimizer": result.get("compiled_optimizer", config.get("compiled_optimizer", False)),
        "options": result.get("options"),
        "cold_timings": {
            "first_update_seconds_including_compile": result.get("first_update_including_compile_seconds"),
            "small_shape_check_seconds_including_compile": numerical.get("seconds_including_small_shape_compile") if numerical else None,
            "optimizer_first_step_seconds_including_compile": optimizer_metadata.get("first_step_seconds_including_compilation"),
            "optimizer_wrapper_setup_seconds": optimizer_metadata.get("compile_wrapper_setup_seconds"),
            "note": "First-invocation wall times include execution and may reuse cached compilations; these are not pure compiler timings",
        },
        "numerical": numerical,
        "optimizer_check": optimizer_check,
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "artifacts": {
            "result": artifact(path),
            "manifest": artifact(manifest_path),
            "numerical_check": artifact(directory / "numerical_check.json"),
            "optimizer_check": artifact(directory / "optimizer_check.json"),
        },
        # Preserve complete measurements/error reports rather than replacing
        # rejected cases with an abbreviated reason or dropping their rows.
        "raw_result": result,
        "manifest": manifest,
    }
    fields = []
    candidate = strict_numbers(candidate, fields)
    candidate["nonfinite_source_fields"] = fields
    return candidate


def is_benchmark_result(path):
    """Identify benchmark experiments by their saved invocation schema."""
    manifest = read_optional(path.parent / "manifest.json")
    config = manifest.get("config") if isinstance(manifest, dict) else None
    return isinstance(config, dict) and "kernels" in config and "steps" in config


def summarize_diagnostic(path):
    """Preserve standalone production/control screens outside candidate counts."""
    result = json.loads(path.read_text())
    manifest_path = path.parent / "manifest.json"
    manifest = read_optional(manifest_path)
    diagnostic = {
        "name": path.parent.name,
        "status": result.get("status", "unknown"),
        "passed": result.get("passed"),
        "numerical_passed": result.get("numerical_passed"),
        "attention": result.get("attention"),
        "shape": result.get("shape"),
        "absolute_loss_difference": result.get("absolute_loss_difference"),
        "global_gradient_relative_l2": result.get("global_gradient_relative_l2"),
        "probe_gradient_relative_l2": result.get("probe_gradient_relative_l2"),
        "thresholds": result.get("thresholds"),
        "controls": {key: value for key, value in result.items() if "control" in key},
        "error": result.get("error"),
        "traceback": result.get("traceback"),
        "artifacts": {"result": artifact(path), "manifest": artifact(manifest_path)},
        "raw_result": result,
        "manifest": manifest,
        "interpretation_note": "A rejected screen alone does not attribute gradient differences to the compiler; assess native repeat controls and deterministic settings in the saved evidence",
    }
    fields = []
    diagnostic = strict_numbers(diagnostic, fields)
    diagnostic["nonfinite_source_fields"] = fields
    return diagnostic


def build_summary(args):
    paths = sorted(args.runs.glob("*/result.json"))
    benchmark_paths = [path for path in paths if is_benchmark_result(path)]
    benchmark_set = set(benchmark_paths)
    candidates = [summarize_candidate(path) for path in benchmark_paths]
    diagnostics = [summarize_diagnostic(path) for path in paths if path not in benchmark_set]
    by_name = {candidate["name"]: candidate for candidate in candidates}
    if args.baseline not in by_name or args.selected not in by_name:
        raise ValueError("Both the baseline and selected candidate must have saved result.json evidence")
    baseline = by_name[args.baseline]
    selected = by_name[args.selected]
    baseline_rate = baseline["tokens_per_second"]
    selected_rate = selected["tokens_per_second"]
    comparable = (baseline["status"] == selected["status"] == "passed"
                  and finite_number(baseline_rate) and baseline_rate > 0
                  and finite_number(selected_rate) and selected_rate > 0)
    pure_seconds = 12 * 60 * 60
    selected_directory = args.runs / args.selected
    evidence = {
        "selected_numerical_check": artifact(selected_directory / "numerical_check.json", include_data=True),
        "selected_optimizer_check": artifact(selected_directory / "optimizer_check.json", include_data=True),
        "flash_attention_evidence": artifact(args.flash_attention_evidence, include_data=True),
        "cuda_batch_equivalence": artifact(args.runs / "cuda_batch_equivalence.json", include_data=True),
        "stable_configuration": artifact(args.backend_config, include_data=True),
    }
    # Keep the earlier artifact key for callers, while naming the chosen
    # reusable configuration neutrally: it can be the throughput-first option.
    evidence["default_configuration"] = evidence["stable_configuration"]
    default_config = evidence["default_configuration"].get("data") or {}
    config_validation = default_config.get("validation_evidence") or {}
    if args.alternative_backend_config:
        evidence["alternative_backend_configurations"] = [artifact(path, include_data=True)
                                                           for path in args.alternative_backend_config]
    if args.pilot_result is not None:
        evidence["optimized_pilot"] = artifact(args.pilot_result, include_data=True)
    if args.nvperf_result is not None:
        evidence["optimized_nvperf"] = artifact(args.nvperf_result, include_data=True)
    if args.production_numerical_result is not None:
        evidence["production_numerical"] = artifact(args.production_numerical_result, include_data=True)
    if args.checkpoint_verification is not None:
        evidence["checkpoint_verification"] = artifact(args.checkpoint_verification, include_data=True)
    if args.pipeline_verification is not None:
        evidence["pipeline_verification"] = artifact(args.pipeline_verification, include_data=True)
    if args.alternative_pipeline_verification:
        evidence["alternative_pipeline_verifications"] = [artifact(path, include_data=True)
                                                          for path in args.alternative_pipeline_verification]
    if args.alternative_pilot_result:
        evidence["alternative_pilots"] = [artifact(path, include_data=True)
                                         for path in args.alternative_pilot_result]
    if args.previous_pilot_result:
        evidence["previous_pilots"] = [artifact(path, include_data=True) for path in args.previous_pilot_result]
    for name in ("native_ulp_sensitivity",):
        linked = config_validation.get(name)
        if linked:
            evidence[name] = artifact(linked, include_data=True)
    production_data = evidence.get("production_numerical", {}).get("data") or {}
    selection_policy = {
        "default_backend_config": str(args.backend_config.resolve()),
        "workflow_validated": default_config.get("validated"),
        "benchmark_status": selected["status"],
        "production_numerical_status": production_data.get("status"),
        "strict_production_numerical_check_passed": config_validation.get("strict_production_numerical_check_passed"),
        "native_gradient_equivalence_claimed": config_validation.get("strict_production_numerical_check_passed") is True,
        "policy": config_validation.get("policy"),
        "selection_context": config_validation.get("selection_context"),
        "purpose": default_config.get("purpose"),
        "production_global_gradient_relative_l2": production_data.get("global_gradient_relative_l2"),
        "production_probe_gradient_relative_l2": production_data.get("probe_gradient_relative_l2"),
        "note": "A passed throughput benchmark or validated workflow configuration does not imply that strict production-shape numerical thresholds passed; saved false/rejected outcomes remain unchanged",
    }
    pilot_estimate = None
    pilot = {}
    if "optimized_pilot" in evidence and evidence["optimized_pilot"]["exists"]:
        pilot = evidence["optimized_pilot"]["data"]
        pilot_rate = pilot.get("steady_tokens_per_second")
        # A completed real training loop provides an additional rate, including
        # data/H2D and loop overhead, rather than silently replacing the short
        # benchmark with a different timing definition.
        pilot_usable = pilot.get("status") == "complete" and finite_number(pilot_rate) and pilot_rate > 0
        pilot_estimate = {
            "theoretical": True,
            "source": "Completed optimized CPT pilot training-loop measurements",
            "source_result": evidence["optimized_pilot"]["path"],
            "pilot_status": pilot.get("status"),
            "pilot_trained_tokens": pilot.get("trained_tokens"),
            "pilot_steady_tokens": pilot.get("steady_tokens"),
            "pilot_steady_seconds": pilot.get("steady_seconds"),
            "pilot_steady_definition": pilot.get("steady_definition"),
            "pilot_train_seconds_including_initial_updates": pilot.get("train_seconds_excluding_validation_and_save"),
            "pilot_tokens_per_second_including_initial_updates": pilot.get("tokens_per_second_including_first_update"),
            "seconds": pure_seconds,
            "steady_tokens_per_second": pilot_rate if pilot_usable else None,
            "tokens": pilot_rate * pure_seconds if pilot_usable else None,
            "billion_tokens": pilot_rate * pure_seconds / 1e9 if pilot_usable else None,
            "excludes": ["data preparation", "model loading", "initial compilation/startup",
                         "validation", "checkpoint save/verification", "restarts and interruptions"],
            "assumption": "The completed pilot's recorded steady-loop rate persists for 12 uninterrupted training hours; extrapolation still excludes whole-workflow overhead",
        }
    pilot_allocated = pilot.get("peak_allocated_bytes")
    pilot_reserved = pilot.get("peak_reserved_bytes")
    memory_measurements = {
        "selected_benchmark_peak_allocated_gib": selected["peak_allocated_gib"],
        "selected_benchmark_peak_reserved_gib": selected["peak_reserved_gib"],
        "completed_pilot_peak_allocated_gib": pilot_allocated / (1024 ** 3) if finite_number(pilot_allocated) else None,
        "completed_pilot_peak_reserved_gib": pilot_reserved / (1024 ** 3) if finite_number(pilot_reserved) else None,
        "pilot_result_path": evidence.get("optimized_pilot", {}).get("path"),
        "scope_note": "The short benchmark uses fixed full batches. The complete pilot's recorded peak includes validation and eager padded-tail/final-partial fallbacks; benchmark memory does not establish the complete-training peak",
    }
    return strict_numbers({
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": artifact(Path(__file__)),
        "runs_directory": str(args.runs.resolve()),
        "candidate_count": len(candidates),
        "candidate_status_counts": dict(sorted(Counter(candidate["status"] for candidate in candidates).items())),
        "production_numeric_screening_count": len(diagnostics),
        "production_numeric_screening_status_counts": dict(sorted(Counter(diagnostic["status"] for diagnostic in diagnostics).items())),
        "production_numeric_screening_evidence": diagnostics,
        "default_selection_validation_policy": selection_policy,
        "baseline": {"name": args.baseline, "status": baseline["status"],
                     "tokens_per_second": baseline_rate, "peak_allocated_gib": baseline["peak_allocated_gib"],
                     "result_path": baseline["artifacts"]["result"]["path"]},
        "selected": {"name": args.selected, "status": selected["status"],
                     "tokens_per_second": selected_rate, "peak_allocated_gib": selected["peak_allocated_gib"],
                     "result_path": selected["artifacts"]["result"]["path"],
                     "speedup_vs_baseline": selected_rate / baseline_rate if comparable else None,
                     "peak_allocated_gib_reduction": baseline["peak_allocated_gib"] - selected["peak_allocated_gib"]
                     if finite_number(baseline["peak_allocated_gib"]) and finite_number(selected["peak_allocated_gib"]) else None},
        "pure_steady_12_hour_token_estimate": {
            "theoretical": True,
            "seconds": pure_seconds,
            "steady_tokens_per_second": selected_rate if comparable else None,
            "tokens": selected_rate * pure_seconds if comparable else None,
            "billion_tokens": selected_rate * pure_seconds / 1e9 if comparable else None,
            "excludes": ["data preparation", "model loading", "compilation and startup",
                         "validation", "checkpoint save/verification", "restarts and interruptions"],
            "assumption": "The measured short-run steady throughput persists for 12 uninterrupted training hours; this is not an end-to-end completion guarantee",
        },
        "actual_pilot_steady_12_hour_token_estimate": pilot_estimate,
        "memory_measurements": memory_measurements,
        "measurement_notes": [
            "Benchmark steady timings exclude first-update compilation/warmup; consult each manifest for exact update counts and effective batch",
            "Candidates with different effective batches are retained for transparency and are not strictly the same training configuration",
            "Numerical checks are saved small-shape reference comparisons; they do not prove identical long-run trajectories",
            "All failed and rejected candidates remain present with complete saved error/numerical evidence",
            "Candidate counts cover only benchmark invocations with manifest.config.kernels and manifest.config.steps; standalone production/control diagnostics are preserved separately",
            "Production-screen rejection requires a suitable native repeat control before assigning its cause to a compiled backend",
            "Default workflow/throughput selection and strict native-gradient validation are separate recorded outcomes; alternatives retain their own validation evidence",
            "Nonfinite numbers from saved evidence become null; nonfinite_source_fields identifies the replacements",
        ],
        "evidence": evidence,
        "candidates": candidates,
    })


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs/cpt_optimization_20261008")
    p.add_argument("--output", type=Path)
    p.add_argument("--baseline", default="native_legacy")
    p.add_argument("--selected", default="compile_precise_compiled_adam")
    p.add_argument("--backend-config", type=Path, default=PROJECT_ROOT / "configs/cpt_backend.json")
    p.add_argument("--flash-attention-evidence", type=Path)
    p.add_argument("--pilot-result", type=Path)
    p.add_argument("--nvperf-result", type=Path)
    p.add_argument("--production-numerical-result", type=Path)
    p.add_argument("--checkpoint-verification", type=Path)
    p.add_argument("--pipeline-verification", type=Path)
    p.add_argument("--alternative-pipeline-verification", type=Path, action="append", default=[])
    p.add_argument("--alternative-pilot-result", type=Path, action="append", default=[])
    p.add_argument("--previous-pilot-result", type=Path, action="append", default=[],
                   help="Preserve an earlier pilot as evidence; may be specified more than once")
    p.add_argument("--alternative-backend-config", type=Path, action="append", default=[],
                   help="Include an optional strict-numerical or memory backend; may be specified more than once")
    return p


def main():
    args = parser().parse_args()
    if args.output is None:
        args.output = args.runs / "summary.json"
    if args.flash_attention_evidence is None:
        args.flash_attention_evidence = args.runs / "flash_attention_evidence.json"
    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "candidates": summary["candidate_count"],
                      "selected": summary["selected"], "theoretical_12h_tokens": summary["pure_steady_12_hour_token_estimate"]["tokens"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
